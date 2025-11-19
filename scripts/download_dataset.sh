if [[ ! -d "dataset" ]]; then
  mkdir dataset
  curl -L -o dataset/70-dog-breedsimage-data-set.zip\
    https://www.kaggle.com/api/v1/datasets/download/gpiosenka/70-dog-breedsimage-data-set
  unzip dataset/70-dog-breedsimage-data-set.zip -d dataset
fi