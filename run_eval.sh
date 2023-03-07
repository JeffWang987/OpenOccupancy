cd $(readlink -f `dirname $0`)
conda activate OpenOccupancy

echo $1
if [ -f $1 ]; then
  config=$1
else
  echo "need a config file"
  exit
fi

export PYTHONPATH="."

ckpt=$2
gpu=$3
bash tools/dist_test.sh $config $ckpt $gpu ${@:4}