import subprocess
import itertools
import time
import os

from time import sleep


def do_test(number_of_netty_threads=1, netty_client_threads=1, default_workers_per_model=1, 
            job_queue_size=100, MKL_NUM_THREADS=1, test_parallelism=8):
    # generate config.properties files based on combination
    config_file_name = "config_file/" + f"config_{number_of_netty_threads}_{netty_client_threads}_{default_workers_per_model}_{job_queue_size}.properties"
    result_file_name = "result_file/" + f"result_{number_of_netty_threads}_{netty_client_threads}_{default_workers_per_model}_{job_queue_size}_{MKL_NUM_THREADS}_{test_parallelism}.txt"

    f = open(config_file_name, "w")
    f.write("load_models=all\n")
    f.write("inference_address=http://0.0.0.0:8080\n")
    f.write("management_address=http://0.0.0.0:8081\n")
    f.write("metrics_address=http://0.0.0.0:8082\n")
    f.write("model_store=<ADD COMPLETE PATH HERE>/model-store\n")
    f.write(f"number_of_netty_threads={number_of_netty_threads}\n")
    f.write(f"netty_client_threads={netty_client_threads}\n")
    f.write(f"default_workers_per_model={default_workers_per_model}\n")
    f.write(f"job_queue_size={job_queue_size}\n")
    f.close()

    # os.system("MKL_NUM_THREADS=1 torchserve --start --model-store model-store --models fast=ptclassifier.mar slow=ptclassifiernotr.mar --ncs --ts-config config.properties")

    # start the torch serve with proper config properties and other parameter settings
    subprocess.call(f"MKL_NUM_THREADS={str(MKL_NUM_THREADS)} torchserve --start --model-store model-store --models model=ptclassifiernotr.mar --ncs --ts-config {config_file_name}", shell=True, stdout=subprocess.DEVNULL)
    sleep(3)

    print(result_file_name)

    # test in parallel to inference API
    print("start to send test request...")
    start_time = time.time()
    print(time.ctime())
    subprocess.run(f"seq 1 1000 | xargs -n 1 -P {str(test_parallelism)} bash -c 'url=\"http://127.0.0.1:8080/predictions/model\"; curl -X POST $url -T input.txt'", shell=True, capture_output=True, text=True)
    total_time = int((time.time() - start_time)*1e6)

    print("total time in ms:", total_time)

    # get metrics of ts inference latency and ts query latency 
    output = subprocess.run("curl http://127.0.0.1:8082/metrics", shell=True, capture_output=True, text=True)

    inference_time=0
    query_time=0
    # capture inference latency and query latency from metrics
    for line in output.stdout.split('\n'):
        if line.startswith('ts_inference_latency_microseconds'):
            inference_time = line.split(' ')[1]
        if line.startswith('ts_queue_latency_microseconds'):
            query_time = line.split(' ')[1]

    # calculate the throughput
    throughput = 1000 / total_time * 1000000

    # write metrics to csv file for display
    f = open("test_result_short.csv", "a")
    f.write(f"{number_of_netty_threads},{netty_client_threads},{default_workers_per_model},{MKL_NUM_THREADS},{job_queue_size},{test_parallelism},{total_time},{inference_time},{query_time},{throughput}\n")
    f.close()

    # stop torchserve for this
    stop_result = os.system("torchserve --stop")
    print(stop_result)
    stop_result = os.system("torchserve --stop")
    print(stop_result)
    stop_result = os.system("torchserve --stop")
    print(stop_result)

def main():
    # set the possible value, value range of each parameter
    number_of_netty_threads = [1, 2, 4, 8]
    netty_client_threads = [1, 2, 4, 8]
    default_workers_per_model = [1, 2, 4, 8]
    MKL_NUM_THREADS = [1, 2, 4, 8]
    job_queue_size = [1000]#[100, 200, 500, 1000]
    test_parallelism = [32]#[8, 16, 32, 64]

    # for each combination of parameters
    [do_test(a, b, c, d, e, f) for a, b, c, d, e, f in itertools.product(number_of_netty_threads, netty_client_threads, default_workers_per_model, job_queue_size, MKL_NUM_THREADS, test_parallelism)]

if __name__ == "__main__":
    main()
