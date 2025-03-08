# Cloud Credentials Setup Guide

This guide explains how to set up AWS and GCP credentials to track real-time AI model costs with the AI Cost Optimizer tool.

## AWS Setup

### 1. Create an AWS IAM User (if you don't already have one)

1. Log in to the AWS Management Console
2. Go to IAM (Identity and Access Management)
3. Navigate to "Users" and click "Add users"
4. Enter a user name and select "Programmatic access"
5. Click "Next: Permissions"
6. Attach the following policies:
   - `AWSCostExplorerServiceFullAccess`
   - `AmazonSageMakerFullAccess` (optional, only if you use SageMaker)
7. Click through to review and create the user
8. **Save the Access Key ID and Secret Access Key** - you'll need these later!

### 2. Configure AWS CLI

1. Install the AWS CLI if you haven't already:
   - Windows: Download from [AWS CLI Installation](https://aws.amazon.com/cli/)
   - Mac/Linux: `pip install awscli`

2. Configure the AWS CLI:
   ```bash
   aws configure
   ```

3. Enter the following information when prompted:
   - AWS Access Key ID: [Your Access Key ID]
   - AWS Secret Access Key: [Your Secret Access Key]
   - Default region name: [Your preferred region, e.g., us-east-1]
   - Default output format: json

### 3. Test AWS Cost Tracking

Run the AWS cost tracker to verify it's working:

```bash
python main.py aws-costs --start-date 2023-01-01 --end-date 2023-12-31
```

## GCP Setup

### 1. Create a Google Cloud Project (if you don't already have one)

1. Go to the [Google Cloud Console](https://console.cloud.google.com/)
2. Click on the project dropdown at the top of the page
3. Click "New Project" and follow the prompts to create a new project

### 2. Enable Billing API

1. In the Google Cloud Console, go to "APIs & Services" > "Library"
2. Search for "Cloud Billing API"
3. Click on "Cloud Billing API" and click "Enable"

### 3. Create a Service Account

1. In the Google Cloud Console, go to "IAM & Admin" > "Service Accounts"
2. Click "Create Service Account"
3. Enter a name for the service account and click "Create"
4. Assign the following roles:
   - Billing Account Viewer
   - Billing Account User
5. Click "Continue" and then "Done"

### 4. Create and Download Service Account Key

1. In the service accounts list, find the service account you just created
2. Click the three dots in the "Actions" column and select "Manage keys"
3. Click "Add Key" > "Create new key"
4. Select JSON as the key type and click "Create"
5. The key file will be downloaded to your computer automatically
6. Move the key file to a secure location on your computer

### 5. Test GCP Cost Tracking

Run the GCP cost tracker to verify it's working:

```bash
python main.py gcp-costs --credentials /path/to/your-gcp-key.json --billing-account YOUR_BILLING_ACCOUNT_ID
```

To find your billing account ID:
1. Go to the [Google Cloud Billing](https://console.cloud.google.com/billing) page
2. Select your billing account
3. The billing account ID appears in the URL and the Account Overview page

## Using Tags/Labels for Model Tracking

To associate costs with specific AI models:

### AWS

Tag your AWS resources (EC2 instances, SageMaker endpoints, etc.) with:
- Key: `ModelName`
- Value: `[Your model name, e.g., bert-base-uncased]`

Then track costs for a specific model:
```bash
python main.py aws-costs --model-name bert-base-uncased
```

### GCP

Label your GCP resources with:
- Key: `model_name`
- Value: `[Your model name, e.g., bert-base-uncased]`

Then track costs for a specific model:
```bash
python main.py gcp-costs --billing-account YOUR_BILLING_ACCOUNT_ID --model-name bert-base-uncased
```

## Security Best Practices

1. **Never commit your credentials to Git repositories**
2. Use environment variables or config files outside your code repo
3. Restrict IAM permissions to only what's necessary
4. Regularly rotate your credentials
5. Consider using AWS roles or GCP workload identity for production deployments 