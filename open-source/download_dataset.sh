set -e

DATASET_NAME="${1}"
OUTPUT_DIR="${2}/${DATASET_NAME}"

BASE_URL="https://storage.googleapis.com/learning-to-simulate-complex-physics/Datasets/${DATASET_NAME}/"

# Create output directory
mkdir -p ${OUTPUT_DIR}

# Download all required files
echo "Downloading dataset files..."
for file in metadata.json train.tfrecord valid.tfrecord test.tfrecord
do
    wget -O "${OUTPUT_DIR}/${file}" "${BASE_URL}${file}"
done

# Transform each split into .pkl format
echo -e "\nTransforming datasets to PKL format..."
for split in train valid test
do
    echo "Processing ${split} split..."
    python transform_to_pkl.py --dataset ${DATASET_NAME} --split ${split}
done

echo -e "\nDownload and transformation completed successfully!"
echo "Dataset location: ${OUTPUT_DIR}"
