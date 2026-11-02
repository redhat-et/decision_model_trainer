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

3. Apply the pipeline storage service:

    ```sh
    oc apply -k openshift/storage/overlays/preprod -n <namespace>
    ```

4. Navigate to the RustFS console through the console route and login with the credentials you created.

## Configuring Pipeline Server Storage

1. Open RHOAI and navigate to "Develop & train"  > "Pipelines" > "Pipeline definitions".
2. Click on "Configure pipeline server".
3. In the RustFS console, click on "Browser", then "+ Create Bucket". Give the bucket a name and click "Create".
4. Click on "Access Keys", then click "+ Add Access Key". Give the key a name and click "Submit".
5. Copy the secret values to the pipeline server configuration dialog, set the URL to the storage API route, and the bucket to the one you created.
6. Submit the form and wait for the pipeline server to start.
