ZIP_FILE=data/CLEVR_v1.0.zip
DATASET_DIR=data/CLEVR_v1.0

if [ -d "$DATASET_DIR" ]; then
    echo "CLEVR dataset already downloaded."
else
    curl https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip -o $ZIP_FILE
    unzip $ZIP_FILE
fi