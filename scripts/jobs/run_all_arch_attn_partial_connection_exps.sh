#!/bin/bash

spaces=("darts" "nb201")
samplers=("darts" "drnas" "gdas" "reinmax")
ks=(2 4 8)
warm_epochs=15

for space in "${spaces[@]}"; do
    for sampler in "${samplers[@]}"; do
        if [ $sampler == "darts" ]; then
            epochs=50
        elif [ $sampler == "drnas" ]; then
            epochs=100
        elif [ $sampler == "gdas" ]; then
            epochs=250
        elif [ $sampler == "reinmax" ]; then
            epochs=250
        else
            echo "Unknown sampler"
            exit 1
        fi

        for k in "${ks[@]}"; do
            echo scripts/jobs/submit_arch_attn_partial_connection_exp.sh $space $sampler $epochs $k $warm_epochs
            sbatch -J ${sampler}-${space}-k${k}-warm_epochs${warm_epochs} scripts/jobs/submit_arch_attn_partial_connection_exp.sh $space $sampler $epochs $k $warm_epochs
        done

    done
done

