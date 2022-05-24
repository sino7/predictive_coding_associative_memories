DATA_DIR=data
ZIP_FILE=data/CLEVR_v1.0.zip
DATASET_DIR=data/CLEVR_v1.0

mkdir $DATA_DIR

if [ -d "$DATASET_DIR" ]; then
    echo "CLEVR dataset already downloaded."
else
    curl https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip -o $ZIP_FILE
    unzip $ZIP_FILE
fi