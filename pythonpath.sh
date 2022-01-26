function join() {
    local IFS=$1
    shift
    echo "$*"
}
mystring=$(join ':' $PWD/*)

export PYTHONPATH="$mystring:$PYTHONPATH"
echo $PYTHONPATH

#export WANDB_API_KEY="0a2ae01d4ea2b07b7fca1f71e45562ab1a123c80"