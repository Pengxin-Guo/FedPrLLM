import argparse
import os 
import numpy as np
import torch
import random
from transformers import AutoTokenizer, AutoModelForCausalLM
from importlib.metadata import version

from lib.prune import prune_wanda, check_sparsity, find_layers, prune_wanda_and_update_global_mask, prune_fedprllm_layer, prune_fedprllm_row_or_column, prune_fedprllmv2
from lib.eval import eval_ppl, eval_zero_shot
from lib.data import get_loaders

print('torch', version('torch'))
print('transformers', version('transformers'))
print('accelerate', version('accelerate'))
print('# of gpus: ', torch.cuda.device_count())

def get_llm(model_name, cache_dir="llm_weights"):
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16, 
        cache_dir=cache_dir, 
        low_cpu_mem_usage=True, 
        device_map="auto"
    )

    model.seqlen = model.config.max_position_embeddings 
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='LLaMA model')
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration samples.')
    parser.add_argument('--sparsity_ratio', type=float, default=0, help='Sparsity level')
    parser.add_argument("--sparsity_type", type=str, default="unstructured", choices=["unstructured"])
    parser.add_argument("--prune_method", type=str, default="wanda", choices=["wanda"])
    parser.add_argument("--cache_dir", default="llm_weights", type=str)
    parser.add_argument('--use_variant', action="store_true", help="whether to use the wanda variant")
    parser.add_argument('--save', type=str, default=None, help='Path to save results.')
    parser.add_argument('--save_model', type=str, default=None, help='Path to save the pruned model.')
    
    parser.add_argument("--method", type=str, default="fedprllm", choices=["centralized", "local", "fedprllm", "fedprllmv2"])    # added by us, for FedPrLLM
    parser.add_argument("--client_num", type=int, default=64)    # added by us, for FedPrLLM
    parser.add_argument("--comparison_group", type=str, default="layer", choices=["layer", "row", "column"])    # added by us, for FedPrLLM
    parser.add_argument('--weight_scaling', action="store_true", help="whether scaling the model weight or not")    # added by us, for FedPrLLM
    args = parser.parse_args()

    # Setting seeds for reproducibility
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    # Handling n:m sparsity
    prune_n, prune_m = 0, 0
    if args.sparsity_type != "unstructured":
        assert args.sparsity_ratio == 0.5, "sparsity ratio must be 0.5 for structured N:M sparsity"
        prune_n, prune_m = map(int, args.sparsity_type.split(":"))

    model_name = args.model.split("/")[-1]
    weight_path = args.cache_dir + model_name + ".pth"
    print(f"loading llm model {args.model}")
    model = get_llm(args.model, args.cache_dir)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)

    device = torch.device("cuda:0")
    if "30b" in args.model or "65b" in args.model: # for 30b and 65b we use device_map to load onto multiple GPUs, thus the processing here.
        device = model.hf_device_map["lm_head"]
    print("use device ", device)
    
    # added by us, move loading data to here
    print("loading calibdation data")
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)
    print("dataset loading complete")
    # end of loading data
    
    if args.method == "centralized":
        print("centralized pruning")
        if args.sparsity_ratio != 0:
            print("centralized pruning starts")
            if args.prune_method == "wanda":
                prune_wanda(args, model, tokenizer, dataloader, device, prune_n=prune_n, prune_m=prune_m)
            else:
                raise NotImplementedError(f"prune method {args.prune_method} not implemented")

        ################################################################
        print("*"*30)
        sparsity_ratio = check_sparsity(model)
        print(f"sparsity sanity check {sparsity_ratio:.4f}")
        print("*"*30)
        ################################################################
        # ppl_test = eval_ppl(args, model, tokenizer, device)
        ppl_test_wiki = eval_ppl(args, model, tokenizer, device, dataset='wikitext2')
        ppl_test_c4 = eval_ppl(args, model, tokenizer, device, dataset='c4')
        ppl_test_ptb = eval_ppl(args, model, tokenizer, device, dataset='ptb')
        print(f"wikitext2 perplexity {ppl_test_wiki}")
        print(f"c4 perplexity {ppl_test_c4}")
        print(f"ptb perplexity {ppl_test_ptb}")

        if not os.path.exists(args.save):
            os.makedirs(args.save)
        save_filepath = os.path.join(args.save, f"log_{args.method}_ns_{args.nsamples}.txt")
        with open(save_filepath, "w") as f:
            print("method\tdataset\tactual_sparsity\tppl_test", file=f, flush=True)
            # print(f"{args.method}\t{sparsity_ratio:.4f}\t{ppl_test:.4f}", file=f, flush=True)
            print(f"{args.method}\twikitext2\t{sparsity_ratio:.4f}\t{ppl_test_wiki:.4f}", file=f, flush=True)
            print(f"{args.method}\tc4\t{sparsity_ratio:.4f}\t{ppl_test_c4:.4f}", file=f, flush=True)
            print(f"{args.method}\tptb\t{sparsity_ratio:.4f}\t{ppl_test_ptb:.4f}", file=f, flush=True)

        if args.save_model:
            model.save_pretrained(args.save_model)
            tokenizer.save_pretrained(args.save_model)
    elif args.method == "local":
        print("local-only pruning")
        torch.save(model.state_dict(), weight_path)    # store original model weight
        client_num, data_sample = args.client_num, args.nsamples // args.client_num
        print("divide data into ", client_num, " clients")
        random.shuffle(dataloader)
        local_dataloaders = [dataloader[i*data_sample:(i+1)*data_sample] for i in range(client_num)]
        print("local training data divided: ", len(local_dataloaders), " clients,", 'each has ', len(local_dataloaders[0]), " samples")
        
        local_ppls_wiki = []
        local_ppls_c4 = []
        local_ppls_ptb = []
        sparsity_ratios = []
        for c in range(client_num):
            dataloader = local_dataloaders[c]
            if args.sparsity_ratio != 0:
                print("local pruning starts, client ", c + 1, "of ", client_num)
                if args.prune_method == "wanda":
                    prune_wanda(args, model, tokenizer, dataloader, device, prune_n=prune_n, prune_m=prune_m)
                else:
                    raise NotImplementedError(f"prune method {args.prune_method} not implemented")
        
            ################################################################
            print("*"*30)
            sparsity_ratio = check_sparsity(model)
            print(f"sparsity sanity check {sparsity_ratio:.4f}")
            print("*"*30)
            sparsity_ratios.append(sparsity_ratio)
            ################################################################
            # ppl_test = eval_ppl(args, model, tokenizer, device)
            ppl_test_wiki = eval_ppl(args, model, tokenizer, device, dataset='wikitext2')
            ppl_test_c4 = eval_ppl(args, model, tokenizer, device, dataset='c4')
            ppl_test_ptb = eval_ppl(args, model, tokenizer, device, dataset='ptb')
            print(f"wikitext2 perplexity {ppl_test_wiki}")
            print(f"c4 perplexity {ppl_test_c4}")
            print(f"ptb perplexity {ppl_test_ptb}")
            local_ppls_wiki.append(ppl_test_wiki)
            local_ppls_c4.append(ppl_test_c4)
            local_ppls_ptb.append(ppl_test_ptb)
            
            model.load_state_dict(torch.load(weight_path))     # recover model weight to original state
        
        print("local pruning complete")
        print("local ppls wikitext2: ", local_ppls_wiki)
        print("average local ppl wikitext2: ", np.mean(local_ppls_wiki))
        print("local ppls c4: ", local_ppls_c4)
        print("average local ppl c4: ", np.mean(local_ppls_c4))
        print("local ppls ptb: ", local_ppls_ptb)
        print("average local ppl ptb: ", np.mean(local_ppls_ptb))
        
        if not os.path.exists(args.save):
            os.makedirs(args.save)
        save_filepath = os.path.join(args.save, f"log_{args.method}__ns_{args.nsamples}_nc_{args.client_num}.txt")
        with open(save_filepath, "w") as f:
            print("method\tdataset\tactual_sparsity\tppl_test", file=f, flush=True)
            # print(f"{args.method}\t{np.mean(sparsity_ratios):.4f}\t{np.mean(local_ppls):.4f}", file=f, flush=True)
            print(f"{args.method}\twikitext2\t{np.mean(sparsity_ratios):.4f}\t{np.mean(local_ppls_wiki):.4f}", file=f, flush=True)
            print("local ppls wikitext2: ", local_ppls_wiki, file=f, flush=True)
            print(f"{args.method}\tc4\t{np.mean(sparsity_ratios):.4f}\t{np.mean(local_ppls_c4):.4f}", file=f, flush=True)
            print("local ppls c4: ", local_ppls_c4, file=f, flush=True)
            print(f"{args.method}\tptb\t{np.mean(sparsity_ratios):.4f}\t{np.mean(local_ppls_ptb):.4f}", file=f, flush=True)
            print("local ppls ptb: ", local_ppls_ptb, file=f, flush=True)
    
    elif args.method == "fedprllm":
        print("fedprllm one-shot pruning")
        torch.save(model.state_dict(), weight_path)    # store original model weight
        client_num, data_sample = args.client_num, args.nsamples // args.client_num
        print("divide data into ", client_num, " clients")
        random.shuffle(dataloader)
        local_dataloaders = [dataloader[i*data_sample:(i+1)*data_sample] for i in range(client_num)]
        print("local training data divided: ", len(local_dataloaders), " clients,", 'each has ', len(local_dataloaders[0]), " samples")
        
        # initialize global mask matrices to zeros
        global_masks = {}
        layers = model.model.layers
        for i in range(len(layers)):
            layer = layers[i]
            subset = find_layers(layer)
            for name in subset:
                unique_name = "layer_" + str(i) + "." + name
                global_masks[unique_name] = torch.zeros(subset[name].weight.data.shape, dtype=torch.int8, device=subset[name].weight.data.device)   # use int type to store mask        
        # end of initialization global mask matrices
        
        # update global mask matrices
        for c in range(client_num):
            dataloader = local_dataloaders[c]
            if args.sparsity_ratio != 0:
                print("use client ", c + 1, "of ", client_num, " to update global mask matrices")
                if args.prune_method == "wanda":
                    prune_wanda_and_update_global_mask(args, model, global_masks, tokenizer, dataloader, device, prune_n=prune_n, prune_m=prune_m)
                else:
                    raise NotImplementedError(f"prune method {args.prune_method} not implemented")
                
                model.load_state_dict(torch.load(weight_path))    # recover model weight to original state
        # end of updating global mask matrices
        
        # prune model at server
        if args.sparsity_ratio != 0:
            print("prune model at server")
            if args.comparison_group == "layer":
                prune_fedprllm_layer(args, model, global_masks, device, prune_n=prune_n, prune_m=prune_m)
            elif args.comparison_group == "row":
                prune_fedprllm_row_or_column(args, model, global_masks, device, dim=1, prune_n=prune_n, prune_m=prune_m)
            elif args.comparison_group == "column":
                prune_fedprllm_row_or_column(args, model, global_masks, device, dim=0, prune_n=prune_n, prune_m=prune_m)
            else:
                raise NotImplementedError(f"comparison groud {args.comparison_group} not implemented")   
        # end of pruning model at server
        
        ################################################################
        print("*"*30)
        sparsity_ratio = check_sparsity(model)
        print(f"sparsity sanity check {sparsity_ratio:.4f}")
        print("*"*30)
        ################################################################
        # ppl_test = eval_ppl(args, model, tokenizer, device)
        ppl_test_wiki = eval_ppl(args, model, tokenizer, device, dataset='wikitext2')
        ppl_test_c4 = eval_ppl(args, model, tokenizer, device, dataset='c4')
        ppl_test_ptb = eval_ppl(args, model, tokenizer, device, dataset='ptb')
        print(f"wikitext2 perplexity {ppl_test_wiki}")
        print(f"c4 perplexity {ppl_test_c4}")
        print(f"ptb perplexity {ppl_test_ptb}")

        if not os.path.exists(args.save):
            os.makedirs(args.save)
        save_filepath = os.path.join(args.save, f"log_{args.method}_ns_{args.nsamples}.txt")
        with open(save_filepath, "w") as f:
            print("method\tdataset\tactual_sparsity\tppl_test", file=f, flush=True)
            # print(f"{args.method}\t{sparsity_ratio:.4f}\t{ppl_test:.4f}", file=f, flush=True)
            print(f"{args.method}\twikitext2\t{sparsity_ratio:.4f}\t{ppl_test_wiki:.4f}", file=f, flush=True)
            print(f"{args.method}\tc4\t{sparsity_ratio:.4f}\t{ppl_test_c4:.4f}", file=f, flush=True)
            print(f"{args.method}\tptb\t{sparsity_ratio:.4f}\t{ppl_test_ptb:.4f}", file=f, flush=True)

        if args.save_model:
            model.save_pretrained(args.save_model)
            tokenizer.save_pretrained(args.save_model)
    
    elif args.method == "fedprllmv2":
        print("fedprllm iterative pruning")
        # torch.save(model.state_dict(), weight_path)    # store original model weight
        client_num, data_sample = args.client_num, args.nsamples // args.client_num
        print("divide data into ", client_num, " clients")
        random.shuffle(dataloader)
        local_dataloaders = [dataloader[i*data_sample:(i+1)*data_sample] for i in range(client_num)]
        print("local training data divided: ", len(local_dataloaders), " clients,", 'each has ', len(local_dataloaders[0]), " samples")
        
        if args.sparsity_ratio != 0:
            print("pruning starts")
            if args.prune_method == "wanda":
                prune_fedprllmv2(args, model, tokenizer, local_dataloaders, client_num, data_sample, device, prune_n=prune_n, prune_m=prune_m)
            else:
                raise NotImplementedError(f"prune method {args.prune_method} not implemented")

        ################################################################
        print("*"*30)
        sparsity_ratio = check_sparsity(model)
        print(f"sparsity sanity check {sparsity_ratio:.4f}")
        print("*"*30)
        ################################################################
        # ppl_test = eval_ppl(args, model, tokenizer, device)
        ppl_test_wiki = eval_ppl(args, model, tokenizer, device, dataset='wikitext2')
        ppl_test_c4 = eval_ppl(args, model, tokenizer, device, dataset='c4')
        ppl_test_ptb = eval_ppl(args, model, tokenizer, device, dataset='ptb')
        print(f"wikitext2 perplexity {ppl_test_wiki}")
        print(f"c4 perplexity {ppl_test_c4}")
        print(f"ptb perplexity {ppl_test_ptb}")

        if not os.path.exists(args.save):
            os.makedirs(args.save)
        save_filepath = os.path.join(args.save, f"log_{args.method}_ns_{args.nsamples}.txt")
        with open(save_filepath, "w") as f:
            print("method\tdataset\tactual_sparsity\tppl_test", file=f, flush=True)
            # print(f"{args.method}\t{sparsity_ratio:.4f}\t{ppl_test:.4f}", file=f, flush=True)
            print(f"{args.method}\twikitext2\t{sparsity_ratio:.4f}\t{ppl_test_wiki:.4f}", file=f, flush=True)
            print(f"{args.method}\tc4\t{sparsity_ratio:.4f}\t{ppl_test_c4:.4f}", file=f, flush=True)
            print(f"{args.method}\tptb\t{sparsity_ratio:.4f}\t{ppl_test_ptb:.4f}", file=f, flush=True)
        
        if args.save_model:
            model.save_pretrained(args.save_model)
            tokenizer.save_pretrained(args.save_model)

    else:
        raise NotImplementedError(f"method {args.method} not implemented")
            
if __name__ == '__main__':
    main()