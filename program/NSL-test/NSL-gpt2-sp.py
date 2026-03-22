import numpy as np
import torch
import time
import math
torch.set_printoptions(8)
#
def params_to_torch(params):
    """递归将字典中的 numpy 数组转换为 torch Tensor"""
    for k, v in params.items():
        if isinstance(v, dict):
            params_to_torch(v)
        elif isinstance(v, list):
            # 针对 blocks 列表
            for item in v:
                if isinstance(item, dict):
                    params_to_torch(item)
        elif isinstance(v, np.ndarray):
            # 转换为 Tensor，并建议转为 float32 避免双精度导致的计算缓慢或冲突
            params[k] = torch.from_numpy(v).float()
    return params

def gelu(x):
    """
        Task: Use the torch API to implement the approximate calculation formula of the `GELU`
        activation function. The formula is as follows (you need to paste it into the latex
        online conversion website)
        Website: https://www.latexlive.com/
        Formula: \frac{1}{2} x\left[1+\tanh \left(\sqrt{\frac{2}{\pi}}\left(x+0.044715 x^{3}\right)\right)\right]
        
        Input: Tensor
        Output: Tensor
    """
    #实现激活函数gelu 其实问题是为什么transformer使用的是gelu而不是relu 这个函数gradient更好些
    arg1 = torch.tanh((x+0.044715* torch.pow(x,3))*math.sqrt(2 / math.pi))
    return 0.5 * x * (arg1 + 1)


def softmax(x):
    """
        Task: Use torch API to implement `softmax` function, search the specific formula by yourself
        Input: Tensor
        Output: Tensor
    """
    #直接调包
    return torch.softmax(x, dim=-1)


def layer_norm(x, g_b, eps:float = 1e-5):
    """
        Task: Use torch API to implement `layernorm` function, search `layernorm` by yourself
        Input: 
            x: Tensor
            g_b: dictionary that load from gpt2 weight. g-gamma and b-bias are the keys
        Output: Tensor
    """
    g, b = torch.Tensor(g_b['g']), torch.Tensor(g_b['b'])
    mean = x.mean(dim = -1, keepdim = True)
    var = x.var(dim = -1,keepdim = True)
    #Norm
    x_norm = (x-mean) / torch.sqrt(var + eps) #eps 防止分母为0
    return x_norm * g + b

def linear(x, w_b):  # [m, in], [in, out], [out] -> [m, out]
    """
        Task: implement linear layer 
        Input: 
            x: Tensor
            w_b: dictionary that load from gpt2 weight. w-weight and b-bias are the keys
        Output: Tensor
    """
    #全连接
    w, b = w_b['w'], w_b['b']
    return torch.matmul(x ,w) + b
    

def ffn(x, mlp):  # [n_seq, n_embd] -> [n_seq, n_embd]
    """
        Task: use `gelu` `linear` to implement ffn
        Notes: x --linear--> --gelu--> --linear--> output
        Input: 
            x: Tensor
            mlp: dictionary that load from gpt2 weight. w_b1 and w_b2 are the params of two linear layer
        Output: Tensor
    """
    w_b1, w_b2 = mlp['c_fc'], mlp['c_proj']
    #linear
    linear1_output = linear(x, w_b1)
    gelu_output = gelu(linear1_output)
    linear2_output = linear(gelu_output, w_b2)
    return linear2_output


def attention(q, k, v, mask):  # [n_q, d_k], [n_k, d_k], [n_k, d_v], [n_q, n_k] -> [n_q, d_v]
    """
        Task: use torch API to implement attention computation according to formula(1) of the following paper
              where d_k account for the last dimension of `k`
        Paper: https://arxiv.org/abs/1706.03762
        Input: 
            q: Tensor
            k: Tensor
            v: Tensor
            mask: Tensor
            mlp: dictionary that load from gpt2 weight. w_b1 and w_b2 are the params of two linear layer
        Output: Tensor
    """
    d_k = k.size(-1) #最后一个dimension是d_model
    #这里实现self-attention 计算部分 其中mask应该是可选添加
    #K 需要先转置  
    scores = torch.matmul(q, k.transpose(-2,-1)) / math.sqrt(d_k)
    #看是否需要mask
    #if mask == True :
        #加一个右上角矩阵就行 softmax之后会无限趋近于0
        #scores = scores.masked_fill(mask == 0, -1e9)
    scores = torch.add(scores, mask)
    #进softmax
    softmax_output = softmax(scores)
    #输出 multiply value即可
    return torch.matmul(softmax_output, v)


def mha(x, attn, n_head):  # [n_seq, n_embd] -> [n_seq, n_embd]
    """
        Task: Complete the code of the multi-head attention
        
        Input: 
            x: Tensor
            attn: dictionary that load from gpt2 weight. c_attn and c_proj are the params of two linear layer
            n_head: number of head
        Output: Tensorying multi-head attention and linear transformation, shape [n_seq, n_embd].
    """
    c_attn, c_proj = attn['c_attn'], attn['c_proj']
    # qkv projection
    x = linear(x, c_attn)  # [n_seq, n_embd] -> [n_seq, 3*n_embd]
    
    # Split into qkv
    """
        Task: Split the q,k,v matrix from the tensor x
        Notes: [n_seq, 3*n_embd] -> 3 * [n_seq, n_embd]
    """
    #其实这里就是projection之后 需要把分出来qkv
    #qkv = None # need to modify
    #直接使用split 沿着最后一个dim分就行
    n_embd = x.size(-1) // 3
    qkv = torch.split(x, n_embd ,dim= -1)

    # Split into heads
    qkv_heads = [qkv_part.chunk(n_head, dim=-1) for qkv_part in qkv]  # 3 * [n_seq, n_embd] -> 3 * n_head * [n_seq, n_embd/n_head]
    qkv_heads = list(zip(*qkv_heads))  # [3, n_head, n_seq, n_embd/n_head]

    # Causal mask to hide future inputs from being attended to
    """
        Task: Construct mask matrix
        Notes: 
            | 0  -inf -inf ... -inf |
            | 0    0  -inf ... -inf |
            | 0    0    0  ... -inf |
            |...  ...  ... ...  ... | 
            | 0    0    0  ...   0  |
        Mask is a tensor whose dimension is [n_seq, n_seq]
    """
    #好吧，直接在这里实现
    #causal_mask = None # need to modify
    n_seq = x.size(-2)
    causal_mask = torch.full((n_seq,n_seq), float('-inf'))
    causal_mask = torch.triu(causal_mask, diagonal=1)
    # Perform attention over each head
    out_heads = [attention(q, k, v, causal_mask) for q, k, v in qkv_heads]  # n_head * [n_seq, n_embd/n_head]
    
    # Merge heads
    """
        Task: merge multi-heads results
        Notes: n_head * [n_seq, n_embd/n_head] --> [n_seq, n_embd]
    """
    #直接拼接在一起即可
    x = torch.cat(out_heads, dim=-1)
    
    # Out projection
    x = linear(x, c_proj)  # [n_seq, n_embd] -> [n_seq, n_embd]
    
    return x


def transformer_block(x, block, n_head):  # [n_seq, n_embd] -> [n_seq, n_embd]
    mlp, attn, ln_1, ln_2 = block['mlp'], block['attn'], block['ln_1'], block['ln_2']
    
    # multi-head causal self attention
    x = x + mha(layer_norm(x, ln_1), attn, n_head=n_head)  # [n_seq, n_embd] -> [n_seq, n_embd]

    # position-wise feed forward network
    x = x + ffn(layer_norm(x, ln_2), mlp)  # [n_seq, n_embd] -> [n_seq, n_embd]

    return x


def gpt2(inputs, params, n_head):  # [n_seq] -> [n_seq, n_vocab]
    wte, wpe, blocks, ln_f = params['wte'], params['wpe'], params['blocks'], params['ln_f']
    # token + positional embeddings
    x = wte[inputs] + wpe[range(len(inputs))]  # [n_seq] -> [n_seq, n_embd]
    
    x = torch.Tensor(x)
    # forward pass through n_layer transformer blocks
    for block in blocks:
        x = transformer_block(x, block, n_head=n_head)  # [n_seq, n_embd] -> [n_seq, n_embd]

    # projection to vocab
    x = layer_norm(x, ln_f)  # [n_seq, n_embd] -> [n_seq, n_embd]
    return x @ wte.T  # [n_seq, n_embd] -> [n_seq, n_vocab]


def generate(inputs, params, n_head, n_tokens_to_generate):
    from tqdm import tqdm

    for _ in tqdm(range(n_tokens_to_generate), "generating"):  # auto-regressive decode loop
        logits = gpt2(inputs, params, n_head=n_head)  # model forward pass
        next_id = np.argmax(logits[-1])  # greedy sampling
        inputs.append(int(next_id))  # append prediction to input

    return inputs[len(inputs) - n_tokens_to_generate :]  # only return generated ids

def greedy_speculative_generate(inputs, draft_params, target_params, hparams_draft, hparams_target, n_tokens_to_generate, K):
    
    """
        Task: Load 124M and 1558M models at the same time, use greedy sampling, and complete speculative decoding
    
        Inputs:
            inputs (list): The initial list of token IDs from the prompt. 
            draft_params, target_params: Model weights for the draft and target models.
            hparams_draft, hparams_target: Hyperparameters for both models.
            n_tokens_to_generate (int): The number of new tokens to generate.
            K (int): The number of tokens the draft model speculates at each step (e.g., 4).

        Returns:
            list: A list of newly generated token IDs.
            
    """
    #先load入两个模型的参数和params
    draft_params = params_to_torch(draft_params)
    target_params = params_to_torch(target_params)
    results = []
    while len(results) < n_tokens_to_generate   :
        #添加一个inputs_draft 用于draft模型的输入 这样就可以避免直接修改inputs 这样就不需要rollback了
        inputs_draft = list(inputs) #每轮都用当前的输入去draft模型预测
        output_ids_draft = generate(inputs_draft, draft_params, hparams_draft['n_head'], K)
        #将draft预测的token添加到输入中
        inputs_draft += output_ids_draft
        #用target模型计算输入的logits 这里forward一遍即可
        logits_target = gpt2(inputs_draft, target_params, hparams_target['n_head'])
        #比较target模型的logits和draft模型的预测 确定接受才加入inputs 
        for i in range(K):
            if np.argmax(logits_target[-(K+1)+i]) != output_ids_draft[i]:
                #使用Greedy Sampling：如果不一致 就把target模型预测概率最大的词添加到inputs中 
                inputs.append(int(np.argmax(logits_target[-K+i])))
                results.append(int(np.argmax(logits_target[-K+i])))
                break
            else:
                #如果一致 就把draft模型预测的token添加到输入中 继续下一轮预测
                inputs.append(output_ids_draft[i])
                results.append(output_ids_draft[i])
                if len(results) >= n_tokens_to_generate:
                    break
        #Bonus Token: 如果draft模型的预测完全被target模型接受 那么还可以把target模型预测的直接加入inputs中 这样就不需要等到下一轮了 直接在当前轮就可以生成更多的token
        if len(results) < n_tokens_to_generate and np.argmax(logits_target[-2]) == output_ids_draft[-1]:
            inputs.append(int(np.argmax(logits_target[-1])))
            results.append(int(np.argmax(logits_target[-1])))   
    #generated_ids = []
    #current_inputs = list(inputs)  
    #return generated_ids
    return results[:n_tokens_to_generate]


def main(prompt: str, n_tokens_to_generate: int = 5, model_size: str = "124M", models_dir: str = "models"):
    from utils import load_encoder_hparams_and_params

    # load encoder, hparams, and params from the released open-ai gpt-2 files
    encoder, hparams, params = load_encoder_hparams_and_params(model_size, models_dir)
    draft_model_size = "124M"
    target_model_size = "1558M"
    encoder, hparams_draft, draft_params = load_encoder_hparams_and_params(draft_model_size, models_dir)
    _, hparams_target, target_params = load_encoder_hparams_and_params(target_model_size, models_dir)
    #转换
    #params = params_to_torch(params)
    # encode the input string using the BPE tokenizer
    input_ids = encoder.encode(prompt)
    # make sure we are not surpassing the max sequence length of our model
    assert len(input_ids) + n_tokens_to_generate < hparams["n_ctx"]

    # generate output ids
    start = time.time()
    #实现投机sampling 
    output_ids = greedy_speculative_generate(input_ids, draft_params, target_params, hparams_draft, hparams_target, n_tokens_to_generate, K=1)
    #output_ids = generate(input_ids, params, hparams["n_head"], n_tokens_to_generate)
    end = time.time()
    print(f"Time taken to generate {n_tokens_to_generate} tokens: {end - start:.2f}s")

    # decode the ids back into a string
    output_text = encoder.decode(output_ids)
    return output_text


if __name__ == "__main__":
    import fire
    fire.Fire(main)