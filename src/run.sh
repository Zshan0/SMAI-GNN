dataset=PROTEINS
name=GIN
folder_name="$name-$dataset"
mkdir $folder_name
for fold_idx in {0..9}
do
  python3 run.py --dataset $dataset --epochs 20 --fold_idx $fold_idx --output_folder $folder_name --name $name --mlp_layers 5
done
zip -r "$folder_name.zip" $folder_name
# from google.colab import files
# files.download("$folder_name.zip")