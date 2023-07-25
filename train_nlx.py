import torch
import torch.nn.functional as F
import torch.utils.data
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoConfig # GPT2Config
from transformers import AdamW, get_linear_schedule_with_warmup
import json
from PIL import Image
from clip_model import CLIPEncoder
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def change_requires_grad(model, req_grad):
    for p in model.parameters():
        p.requires_grad = req_grad


def load_checkpoint(ckpt_path, epoch):
    
    model_name = 'unified_nle_model_{}'.format(str(epoch))
    tokenizer_name = 'unified_nle_tokenizer_0'
    filename = 'ckpt_stats_' + str(epoch) + '.tar'
    
    tokenizer = GPT2Tokenizer.from_pretrained(ckpt_path + tokenizer_name)        # load tokenizer
    model = GPT2LMHeadModel.from_pretrained(ckpt_path + model_name).to(device)   # load model with config
    opt = torch.load(ckpt_path + filename)
    optimizer = get_optimizer(model, learning_rate)
    optimizer.load_state_dict(opt['optimizer_state_dict'])
    start_epoch = opt['epoch'] + 1
    scheduler_dic = opt['scheduler']
    del opt
    torch.cuda.empty_cache()

    return tokenizer, model, optimizer, scheduler_dic, start_epoch

def load_pretrained():
    
    model_path = 'pretrained_model/pretrain_model_14'
    tokenizer_path = 'pretrained_model/pretrain_tokenizer_0'
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)        # load tokenizer
    model = GPT2LMHeadModel.from_pretrained(model_path).to(device)   # load model with config
    return tokenizer, model
    

def save_checkpoint(epoch, model, optimizer, tokenizer, scheduler, ckpt_path, **kwargs):
    
    model_name = 'unified_nle_model_{}'.format(str(epoch))
    tokenizer_name = 'unified_nle_tokenizer_{}'.format(str(epoch))
    filename = 'ckpt_stats_' + str(epoch) + '.tar'
    
    if epoch == 0:
        tokenizer.save_pretrained(ckpt_path + tokenizer_name)   # save tokenizer
        
    model.save_pretrained(ckpt_path + model_name)
        
    opt = {'epoch': epoch,
           'optimizer_state_dict': optimizer.state_dict(), 
           'scheduler': scheduler.state_dict(),
            **kwargs}
    
    torch.save(opt, ckpt_path + filename)
    

class UnifiedTrainDataset(Dataset):

    def __init__(self, nle_path, dataset_base_path, transform, tokenizer, max_seq_len):
        
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_seq_len = max_seq_len       # question + <bos> The answer is <answer> becase <explanation> <eos>
        self.data = json.load(open(nle_path, 'r'))
        self.dataset_base_path = dataset_base_path

    def __getitem__(self, i):
        
        sample = self.data[i]
        
        # extract information
        text_a = sample['question']  # question
        answer = sample['answer']
        text_b = sample['explanation']  # explanation
        img_path = self.dataset_base_path + sample['img_path']
        
        additional_tokens = ['<question>', '<answer>', '<explanation>']

        # tokenization process
        q_segment_id, a_segment_id, e_segment_id = self.tokenizer.convert_tokens_to_ids(additional_tokens)
        tokens = self.tokenizer.tokenize(text_a)
        labels = [-100] * len(tokens)   # we dont want to predict the question, set to pad to ignore in XE
        segment_ids = [q_segment_id] * len(tokens)

        answer = [self.tokenizer.bos_token] + self.tokenizer.tokenize(" the answer is " + answer)
        answer_len = len(answer)
        tokens_b = self.tokenizer.tokenize(" because " + text_b) + [self.tokenizer.eos_token]
        exp_len = len(tokens_b)
        tokens += answer + tokens_b
        labels += [-100] + answer[1:] + tokens_b   # labels will be shifted in the model, so for now set them same as tokens
        segment_ids += [a_segment_id] * answer_len
        segment_ids += [e_segment_id] * exp_len

        if len(tokens) > self.max_seq_len :
            tokens = tokens[:self.max_seq_len]
            labels = labels[:self.max_seq_len]
            segment_ids = segment_ids[:self.max_seq_len]


        assert len(tokens) == len(segment_ids) 
        assert len(tokens) == len(labels)
        
        seq_len = len(tokens)
        padding_len = self.max_seq_len - seq_len
        tokens = tokens + ([self.tokenizer.pad_token] * padding_len)
        labels = labels + ([-100] * padding_len)
        
        segment_ids += ([e_segment_id] * padding_len)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_ids = torch.tensor(input_ids, dtype=torch.long)

        labels = [self.tokenizer.convert_tokens_to_ids(t) if t!=-100 else t for t in labels]
        labels = torch.tensor(labels, dtype=torch.long)
        
        segment_ids = torch.tensor(segment_ids, dtype=torch.long)
        

        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        
        return (img, input_ids, labels, segment_ids)

    def __len__(self):
        return len(self.data)
    

def get_optimizer(model, learning_rate):
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],  
         'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 
         'weight_decay': 0.0}]

    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
    return optimizer


nle_data_train_path = 'datasets/explanation_dataset.json'
dataset_base_path = 'image_datasets/'

img_size = 224
ckpt_path = 'ckpts/'
max_seq_len = 125
load_from_epoch = None
no_sample = True   # setting this to False will greatly reduce the evaluation scores, be careful!
top_k =  0
top_p =  0.9
batch_size = 64   # per GPU 
num_train_epochs = 20
weight_decay = 0
start_epoch = 0
temperature = 1
finetune_pretrained = False
learning_rate = 1e-5 if finetune_pretrained else 2e-5

image_encoder = CLIPEncoder(device)
change_requires_grad(image_encoder, False)


if load_from_epoch is not None:
    tokenizer, model, optimizer, scheduler_dic, start_epoch = load_checkpoint(ckpt_path, load_from_epoch)
else:
    if finetune_pretrained:
        tokenizer, model = load_pretrained()
        optimizer = get_optimizer(model, learning_rate)
    else:
        # tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
        orig_num_tokens = len(tokenizer.encoder)
        num_new_tokens = tokenizer.add_special_tokens({'pad_token': '<pad>',
                                                       'additional_special_tokens': ['<question>', '<answer>', '<explanation>']})
        assert len(tokenizer) == orig_num_tokens + num_new_tokens
        
        # config = GPT2Config()
        config = AutoConfig.from_pretrained('distilgpt2')
        
        # Add configs
        setattr(config, 'img_size', None)
        setattr(config, 'max_seq_len', None)   
        config.img_size = img_size
        config.max_seq_len = max_seq_len 
        config.add_cross_attention = True
        
        # model = GPT2LMHeadModel.from_pretrained('gpt2', config = config)
        model = GPT2LMHeadModel.from_pretrained('distilgpt2', config = config)
        model.resize_token_embeddings(len(tokenizer))
        model = model.to(device)
        optimizer = get_optimizer(model, learning_rate)
    
print("Model Setup Ready...")

img_transform = transforms.Compose([transforms.Resize((img_size,img_size)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

train_dataset = UnifiedTrainDataset(nle_path = nle_data_train_path, 
                                    dataset_base_path = dataset_base_path,
                                    transform = img_transform, 
                                    tokenizer = tokenizer, 
                                    max_seq_len = max_seq_len)

train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size = batch_size, 
                                           shuffle=True, 
                                           pin_memory=True)



t_total = len(train_loader) * num_train_epochs
warmup_steps = 0   # 0.10 * t_total
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)

if load_from_epoch is not None:
    scheduler.load_state_dict(scheduler_dic)


for epoch in range(start_epoch, num_train_epochs):
    
    model.train()
    
    for step, batch in enumerate(train_loader):
        
        
        batch = tuple(input_tensor.to(device) for input_tensor in batch)
        img, input_ids, labels, segment_ids = batch
        
        img_embeddings = image_encoder(img)
        
        outputs = model(input_ids=input_ids, 
                        past_key_values=None, 
                        attention_mask=None, 
                        token_type_ids=segment_ids, 
                        position_ids=None, 
                        encoder_hidden_states=img_embeddings, 
                        encoder_attention_mask=None, 
                        labels=labels, 
                        use_cache=False, 
                        return_dict=True)
        
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        print("\rEpoch {} / {}, Iter {} / {}, Loss: {:.3f}".format(epoch, 
                                                                   num_train_epochs, 
                                                                   step, len(train_loader), 
                                                                   loss.item()), end='          ')
            
            

    save_checkpoint(epoch, model, optimizer, tokenizer, scheduler, ckpt_path)
                                                                                      

    