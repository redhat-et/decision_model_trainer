# Pipeline Server RustFS S3-Compatible Storage

## Setup

1. Copy the storage secret and fill in the values:

    ```sh
    cp openshift/storage/base/secret.example.yaml openshift/storage/base/secret.yaml
    ```

2. Login to your cluster and apply the secret:

    ```sh
    oc login --server=<server> --token=<token>
    oc apply openshift/storage/base/secret.yaml -n <namespace>
    ```

3. Apply the storage service:

    ```sh
    oc apply -k openshift/storage/overlays/preprod -n <namespace>
    ```
