#!/bin/bash

# Command to execute
CMD="/home/huyaoqi/anaconda3/bin/python /root/granite-tsfm/ttm_pretrain_stock.py"

# Loop infinitely
while true; do
    # Execute the command
    $CMD
    # Check the exit status of the command
    if [ $? -eq 0 ]; then
        # Exit the loop if the command was successful
        echo "Command executed successfully."
        break
    else
        # Print an error message and retry after a short delay
        echo "Command failed. Retrying in 5 seconds..."
        sleep 5
    fi
done
