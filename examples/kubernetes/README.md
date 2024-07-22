# optimum-habana-example-chart

![Version: 0.1.0](https://img.shields.io/badge/Version-0.1.0-informational?style=flat-square)

This folder contains a Dockerfile and [Helm chart](https://helm.sh) demonstrating how 🤗 Optimum Habana examples
can be run using Intel® Gaudi® AI accelerator nodes from a vanilla Kubernetes cluster. The instructions below
explain how to build the docker image and deploy the job to a Kubernetes cluster using Helm.

## Requirements

### Client System Requirements

In order to build the docker images and deploy a job to Kubernetes you will need:

* [Docker](https://docs.docker.com/engine/install/)
* [Docker compose](https://docs.docker.com/compose/install/)
* [kubectl](https://kubernetes.io/docs/tasks/tools/) installed and configured to access a cluster
* [Optional] [Minikube](https://minikube.sigs.k8s.io/docs/start/) installed and configured to access a Kubernetes cluster on a single node
* [Helm CLI](https://helm.sh/docs/intro/install/)

> If you are intending to run a Kubernetes cluster on a single node (locally), you will need a tool like [Minikube](https://minikube.sigs.k8s.io/docs/start/) to be installed first

### Cluster Requirements

Your Kubernetes cluster will need the [Intel Gaudi Device Plugin for Kubernetes](https://docs.habana.ai/en/latest/Orchestration/Gaudi_Kubernetes/Device_Plugin_for_Kubernetes.html)
in order to request and utilize the accelerators in the Kubernetes job. Also, ensure that your Kubernetes version is supported based on the
[support matrix](https://docs.habana.ai/en/latest/Support_Matrix/Support_Matrix.html#support-matrix).

## Container

The [Dockerfile](Dockerfile) and [docker-compose.yaml](docker-compose.yaml) build the following images:

* An `optimum-habana` base image that uses the [PyTorch Docker images for Gaudi](https://developer.habana.ai/catalog/pytorch-container/) as it's base, and then installs
optimum-habana and the Habana fork of Deep Speed.
* An `optimum-habana-examples` image is built on top of the `optimum-habana` base to includes installations from
`requirements.txt` files in the example directories and a clone of [this GitHub repository](https://github.com/huggingface/optimum-habana/) in order to run example scripts.

Use the the following commands to build the containers:

> Note that the `GAUDI_SW_VER`, `OS`, and `TORCH_VER` are used to
> determine which [Intel Gaudi PyTorch base image](https://developer.habana.ai/catalog/pytorch-container/) to use. The
> combination of versions provided must match one of the pre-built images that are available in the Intel Gaudi Vault.

```bash
# Specify the Gaudi SW version, OS, and PyTorch version which will be used for the base container
export GAUDI_SW_VER=1.16.2
export OS=ubuntu22.04
export TORCH_VER=2.2.2

# Specify the version of optimum-habana to install in the container
export OPTIMUM_HABANA_VER=1.12.1

git clone https://github.com/huggingface/optimum-habana.git

# Note: Modify the requirements.txt file in the kubernetes directory for the specific example(s) that you want to run
cd optimum-habana/examples/kubernetes

# Set variables for your container registry and repository
export REGISTRY=<Your container registry>
export REPO=<Your container repository>

# Build the images
docker compose build

# Push the optimum-habana-examples image to a container registry
docker push <image name>:<tag>
```

## Helm Chart

### Kubernetes resources

This Kubernetes job uses a [Helm chart](https://helm.sh) with the following resources:
* Job to run the Optimum Habana example script using HPU(s) from a single node
* [Persistant volume claim](https://kubernetes.io/docs/concepts/storage/persistent-volumes/#persistentvolumeclaims)
  (PVC) backed by NFS to store output files
* (Optional) Pod used to access the files from the PVC after the worker pod completes
* (Optional) [Secret](https://kubernetes.io/docs/concepts/configuration/secret/) for a Hugging Face token, if gated
  models are being used

### Helm chart values

The [Helm chart values file](https://helm.sh/docs/chart_template_guide/values_files/) is a yaml file with values that
get passed to the chart when it's deployed to the cluster. These values specify the python script and parameters for
your job, the name and tag of your Docker image, the number of HPU cards to use for the job, etc.

<details>
  <summary> Expand to see the values table </summary>

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| affinity | object | `{}` | Optionally provide node [affinities](https://kubernetes.io/docs/concepts/scheduling-eviction/assign-pod-node/#affinity-and-anti-affinity) to constrain which node your worker pod will be scheduled on. |
| command[0] | string | `"python"` |  |
| command[1] | string | `"/workspace/optimum-habana/examples/gaudi_spawn.py"` |  |
| command[2] | string | `"--help"` |  |
| env | list | `[{"name":"LOGLEVEL","value":"INFO"}]` | Define environment variables to set in the container |
| envFrom | list | `[]` | Optionally define a config map's data as container environment variables |
| hostIPC | bool | `false` | The default 64MB of shared memory for docker containers can be insufficient when using more than one HPU. Setting hostIPC: true allows reusing the host's shared memory space inside the container. |
| image.pullPolicy | string | `"IfNotPresent"` | Determines when the kubelet will pull the image to the worker nodes. Choose from: `IfNotPresent`, `Always`, or `Never`. If updates to the image have been made, use `Always` to ensure the newest image is used. |
| image.repository | string | `nil` | Repository and name of the docker image |
| image.tag | string | `nil` | Tag of the docker image |
| imagePullSecrets | list | `[]` | Optional [image pull secret](https://kubernetes.io/docs/tasks/configure-pod-container/pull-image-private-registry/) to pull from a private registry |
| nodeSelector | object | `{}` | Optionally specify a [node selector](https://kubernetes.io/docs/concepts/scheduling-eviction/assign-pod-node/#nodeselector) with labels the determine which node your worker pod will land on. |
| podAnnotations | object | `{}` | Pod [annotations](https://kubernetes.io/docs/concepts/overview/working-with-objects/annotations/) to attach metadata to the job |
| podSecurityContext | object | `{}` | Specify a pod security context to run as a non-root user |
| resources.limits."habana.ai/gaudi" | int | `1` | Specify the number of Gaudi card(s) |
| resources.limits.cpu | int | `16` | Specify [CPU resource](https://kubernetes.io/docs/concepts/configuration/manage-resources-containers/#meaning-of-cpu) limits for the job |
| resources.limits.hugepages-2Mi | string | `"4400Mi"` | Specify hugepages-2Mi limit for the job |
| resources.limits.memory | string | `"128Gi"` | Specify [memory limits](https://kubernetes.io/docs/concepts/configuration/manage-resources-containers/#meaning-of-memory) requests for the job |
| resources.requests."habana.ai/gaudi" | int | `1` | Specify the number of Gaudi card(s) |
| resources.requests.cpu | int | `16` | Specify [CPU resource](https://kubernetes.io/docs/concepts/configuration/manage-resources-containers/#meaning-of-cpu) requests for the job |
| resources.requests.hugepages-2Mi | string | `"4400Mi"` | Specify hugepages-2Mi requests for the job |
| resources.requests.memory | string | `"128Gi"` | Specify [memory resource](https://kubernetes.io/docs/concepts/configuration/manage-resources-containers/#meaning-of-memory) requests for the job |
| secret.encodedToken | string | `nil` | Hugging Face token encoded using base64. |
| secret.secretMountPath | string | `"/tmp/hf_token"` | If a token is provided, specify a mount path that will be used to set HF_TOKEN_PATH |
| securityContext.privileged | bool | `false` | Run as privileged or unprivileged. Certain deployments may require running as privileged, check with your system admin. |
| storage.accessModes | list | `["ReadWriteMany"]` | [Access modes](https://kubernetes.io/docs/concepts/storage/persistent-volumes/#access-modes) for the persistent volume. |
| storage.deployDataAccessPod | bool | `true` | A data access pod will be deployed when set to true. This allows accessing the data from the PVC after the worker pod has completed. |
| storage.pvcMountPath | string | `"/tmp/pvc-mount"` | Locaton where the PVC will be mounted in the pod |
| storage.resources | object | `{"requests":{"storage":"30Gi"}}` | Storage [resources](https://kubernetes.io/docs/concepts/storage/persistent-volumes/#resources) |
| storage.storageClassName | string | `"nfs-client"` | Name of the storage class to use for the persistent volume claim. To list the available storage classes use: `kubectl get storageclass`. |
| tolerations | list | `[]` | Optionally specify [tolerations](https://kubernetes.io/docs/concepts/scheduling-eviction/taint-and-toleration/) to allow the worker pod to land on a node with a taint. |

</details>

There is a [`values.yaml` files](values.yaml) that is intended to be used as a template to run an optimum-habana
example script. Ensure that your container has the requirements needed to run the example, and then update the
`values.yaml` file with your `image.repository` and `image.tag`. Then, update the `command` array with the script and
parameters to run the example.

Validated use cases can be found in the `ci` directory:

| Values files path | HPUs | Description |
|-------------------|------|-------------|
| [`ci/single-card-glue-values.yaml`](ci/single-card-glue-values.yaml) | 1 | Uses a single card to [fine tune BERT large](../text-classification/README.md#single-card-training) (with whole word masking) on the text classification MRPC task using `run_glue.py`.
| [`ci/multi-card-glue-values.yaml`](ci/multi-card-glue-values.yaml) | 2 | Uses 2 HPUs from a single node with the [`gaudi_spawn.py`](../gaudi_spawn.py) script to [fine tune BERT large](../text-classification/README.md#multi-card-training) (with whole word masking) on the text classification MRPC task using `run_glue.py`.
| [`ci/single-card-lora-clm-values.yaml`](ci/single-card-lora-clm-values.yaml) | 1 | Uses a single card to [fine tune Llama1-7B](../language-modeling/README.md#peft) with LoRA using the `run_lora_clm.py` script.
| [`ci/multi-card-lora-clm-values.yaml`](ci/multi-card-lora-clm-values.yaml) | 8 | Uses 8 HPUs from a single node with the [`gaudi_spawn.py`](../gaudi_spawn.py) script to [fine tune Llama1-7B](../language-modeling/README.md#peft) with LoRA using the `run_lora_clm.py` script.

### Deploy job to the cluster

After updating the values file for the example that you want to run, use the following command to deploy the job to
your Kubernetes cluster.

```bash
cd examples/kubernetes

helm install -f <path to values.yaml> optimum-habana-examples . -n <namespace>
```

After the job is deployed, you can check the status of the pods:
```bash
kubectl get pods -n <namespace>
```

To monitor a running job, you can view the logs of the worker pod:

```bash
kubectl logs <pod name> -n <namespace> -f
```

The data access pod can be used to copy artifacts from the persistent volume claim (for example, the trained model
after fine tuning completes). Note that this requires that the data access pod be deployed to the cluster with the helm
chart by setting `storage.deployDataAccessPod = true` in the values yaml file. The path to the files is defined in the
`storage.pvcMountPath` value (this defaults to `/tmp/pvc-mount`). You can find the name of your data access pod using
`kubectl get pods -n <namespace> | grep dataaccess`.

```bash
# Copy files from the PVC mount path to your local machine
kubectl cp <data access pod name>:/tmp/pvc-mount . -n <namespace>
```

Finally, when your job is complete and you've copied all the artifacts that you need to your local machine, you can
uninstall the helm job from the cluster:

```bash
helm uninstall optimum-habana-examples . -n <namespace>
```

----------------------------------------------
Autogenerated from chart metadata using [helm-docs v1.13.1](https://github.com/norwoodj/helm-docs/releases/v1.13.1)
