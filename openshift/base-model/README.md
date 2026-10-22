# Decision Model Serving

Create the storage secret before applying the model deployment:

```sh
cp openshift/decision-model/base/secret.example.yaml openshift/decision-model/base/secret.yaml
```

Set `MODEL_S3_URI`, `AWS_ENDPOINT_URL`, `AWS_ACCESS_KEY_ID`, and
`AWS_SECRET_ACCESS_KEY` for the S3-compatible storage provider, then apply it
to the namespace:

```sh
oc apply -f openshift/decision-model/base/secret.yaml -n <namespace>
oc apply -k openshift/decision-model/overlays/preprod -n <namespace>
```

Test the endpoint

```sh
MODEL_ROUTE=$(oc get route decision-model-route -o jsonpath='http://{.spec.host}{"\n"}')

curl -s $MODEL_ROUTE/v1/systemone \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "state": {"ticket": "My card was charged twice for one order."},
    "questions": {
      "team": {
        "type": "choice",
        "instructions": "Which team should handle this ticket?",
        "criteria": {
          "billing": "payments and refunds",
          "shipping": "deliveries",
          "security": "account access"
        }
      },
      "urgent": {
        "type": "noul",
        "instructions": "Does the customer need a reply within the hour?",
        "criteria": {"true": "the account is locked or money is moving"}
      },
      "tone": {
        "type": "score",
        "instructions": "How upset is the customer?",
        "criteria": ["calm", "annoyed", "furious"]
      }
    },
    "chat_template_kwargs": {"enable_thinking": false}
  }'
```