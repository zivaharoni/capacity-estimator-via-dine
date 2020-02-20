# Capacity Estimation via Directed Information Neural Estimator (DINE)

This repository contains an implementation of capacity of continuous channels as introduced in (Link will be added in the future).

## Prerequisites

The code is compatible with a tensoreflow 2.0 environment.
If you use a docker, you can pull the following docker image

```
docker pull tensorflow/tensorflow:latest-gpu-py3
```


## Running the code

The estimate the capacity of the AWGN channel run
```
python ./main.py --name <simulation_name> --config_name awgn --P <source_power> --C <capacity_for_visualization> &
```
The estimate the capacity of the MA(1)-AGN channel run
```
python ./main.py --name <simulation_name> --config_name arma_ff --P <source_power> --C <capacity_for_visualization> &
```
The estimate the feedback capacity of the MA(1)-AGN channel run
```
python ./main.py --name <simulation_name> --config_name arma_fb --P <source_power> --C <capacity_for_visualization> &
```
## Authors

* **Ziv Aharoni** 
* **Dor Tsur** 
* **Ziv Goldfeld** 
* **Haim Permuter** 


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

