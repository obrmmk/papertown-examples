import os
import wandb
import torch.distributed as dist

from transformers import DataCollatorForLanguageModeling
from transformers import Trainer

from papertown import DataComposer, load_tokenizer
from papertown.new_model import new_Llama2
from args_parser import ArgsHandler, FileLoader, ModelCreator

IS_DISTRIBUTED = 'WORLD_SIZE' in os.environ and int(os.environ['WORLD_SIZE']) > 1
if IS_DISTRIBUTED:
    dist.init_process_group(backend='nccl', init_method='env://')

args_handler = ArgsHandler()
args, training_args = args_handler.get_args()
urls = FileLoader.load_urls_from_txtfile(args['urls'], args['sample_size'])

try:
    if not IS_DISTRIBUTED or (IS_DISTRIBUTED and dist.get_rank() == 0):
        wandb.init(
            entity= args['entity'], 
            project= args['project'],
            name= args['name']
        )
        wandb.alert (title= "Job Started", text= "動いたよ! Yay!" )

    tokenizer = load_tokenizer()
    model = ModelCreator.create(new_Llama2, args, tokenizer)

    with DataComposer(url_list=urls, format='pre',
                    block_size=args['gradient_accumulation_steps'], prefetch=1) as dataset: 
        data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,
        )

        result = trainer.train()
        output_path = args['trained_dir']
        tokenizer.save_pretrained(output_path)
        model.save_pretrained(output_path)

finally:
    if IS_DISTRIBUTED:
        dist.destroy_process_group()