import torch

import os
import yaml
import numpy as np

import huggingface_hub as hf

import utils.constants as constants

class BaseTrainer:

  _hyper_file = os.path.join(constants.LOCAL_DATA_PATH, "hyperparams.yml")

  def __init__(
    self,
    save_name,
    **kwargs
  ):
    self.save_name = save_name
    self.save_repo = f"{constants.HF_ID}/{save_name}"
    hf.create_repo(
        save_name, private=True, exist_ok=True
    )
    os.makedirs(constants.LOCAL_DATA_PATH, exist_ok=True)

    try:
        h = self._hyperparams
    except:
        raise NotImplementedError("Please define _hyperparams in your trainer!")
    
    for k in h:
        setattr(self, k, kwargs[k])
        

  @torch.no_grad()
  def upload(self, *files):
    api = hf.HfApi()

    # save hyperparams as csv
    with open(self._hyper_file, 'w') as outfile:
      yaml.dump(
        {k: str(getattr(self, k)) for k in self._hyperparams},
        outfile,
        default_flow_style=False
      )

    for file in list(files) + [self._hyper_file]:
        api.upload_file(
            path_or_fileobj=file,
            path_in_repo=str(file).split("/")[-1],
            repo_id=self.save_repo,
            repo_type="model"
        )


  @torch.no_grad()
  def save_checkpoint(
    self,
    models
  ):
    api = hf.HfApi()

    for name, model in models.items():
      model.save_pretrained(
          os.path.join(constants.LOCAL_DATA_PATH, name),
          push_to_hub=False,
      )

      api.upload_folder(
          repo_id=self.save_repo,
          folder_path=os.path.join(constants.LOCAL_DATA_PATH, name),
          path_in_repo=name,
          repo_type="model"
      )
