import json
import os
from typing import List, Union

import torch
from langchain.schema import Document   
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from torch.utils.data import DataLoader, Dataset

from src.parser import simple_text_split
import yaml
from src import CFG

class Propositioner:
    """Based of the https://github.com/chentong0/factoid-wiki"""

    def __init__(self):
        model_path = os.path.join(CFG["MODELS_DIR"], CFG["PROPOSITIONER_PATH"])
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_path, device_map=CFG["DEVICE"]
        )
        self.model.eval()

    def _predict(self, text: Union[str, List[str]]) -> List[str]:
        input_ids = self.tokenizer(text, return_tensors="pt").input_ids
        with torch.no_grad():
            outputs = self.model.generate(input_ids,
                            max_new_tokens=CFG["PROPOSITIONER_CONFIG"]["CHUNK_SIZE"])
        output_text = self.tokenizer.decode(outputs, skip_special_tokens=True)
        return output_text
    
    def generate(
            self, passage: Document, title: str = "", section: str =  "" 
    )-> List[Document]:
        input_text = f"Title: {title}. Section: {section}. Content: {passage.page_content}"
        output_text = self._predict(input_text)
        metadata = passage.metadeta.copy()
        return [
            Document(page_content=x, metadata=metadata) for x in metadata 
        ]
    

    def batch(self, passages: List[Document], title: str = "", section: str = ""
    ) -> List[Document]:
        data_set = DocDataSet(passages, section, title)

        data_loader = DataLoader(
            data_set, batch_size=16, shuffle=True, drop_last=True
        )

        prop_texts = []
        for data in data_loader:
            input_texts, source = data
            output_texts = self._predict(input_texts)

            for output_text, source, input_text in zip(
                output_texts, source, input_texts
            ):
                try:
                    prop_texts.extend(
                        [
                            Document(page_content=x, metadata={"source": source})
                            for x in json.loads(output_text)
                        ]
                    )
                except json.JSONDecodeError:
                    texts = simple_text_split(
                        Document(page_content=input_text, metadata={"source": source}),
                        CFG["PROPOSITIONER_CONFIG"]["CHUNK_SIZE"],
                        CFG["PROPOSITIONER_CONFIG"]["CHUNK_OVER_LAP"]
                    )
                    prop_texts.extend(texts)

        return prop_texts



class DocDataSet(Dataset):
    def __init__(self, passages: List[Document], section: str, title: str):
        self.texts = [
            f"Title: {title}. Section: {section}. Context: {passage.page_content}"
            for passage in passages
        ]
        self.sources = [passage.metadata["source"] for passage in passages]

    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return self.texts[idx], self.source[idx]
