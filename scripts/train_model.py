from transformers import DataCollatorForLanguageModeling
from transformers import Trainer

from papertown import DataComposer, load_tokenizer
from papertown.new_model import new_Llama2

import wandb
from args_parser import ArgsHandler, FileLoader, ModelCreator

args, training_args = ArgsHandler.get_args()
urls = FileLoader.load_urls_from_txtfile(args['urls'])

wandb.init(
    entity= args['entity'], 
    project= args['project'],
    name= args['name']
)
wandb.alert (title= "Job Started", text= "動いたよ! Yay!" )

tokenizer = load_tokenizer()
model = ModelCreator.create(new_Llama2, args, tokenizer)

# Append the sample size to the URL, but only if sample_size is provided (i.e., not -1)
sample_size = args['sample_size']
if sample_size != -1:
    urls = [url + f'[:{sample_size}]' for url in urls]

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

