# 🚀 Supply Chain Optimization System - Deployment Guide

This guide provides step-by-step instructions for deploying the Intelligent Supply Chain Optimization System to both Render.com and AWS.

## 🎯 Render.com Deployment (Recommended)

Render.com provides a simple, scalable platform for deploying web applications with automatic SSL, custom domains, and managed databases.

### Prerequisites

- **Render.com Account** (free tier available)
- **GitHub Repository** connected to your Render account
- **Git** for version control

### Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐
│   Frontend      │    │   Backend API    │
│   (Static Site) │────│   (Web Service)  │
└─────────────────┘    └──────────────────┘
         │                        │
         │                        │
┌─────────────────┐    ┌─────────────────┐
│   PostgreSQL    │    │   Redis Cache   │
│   (Database)    │    │   (Optional)     │
└─────────────────┘    └─────────────────┘
```

### Quick Deployment

1. **Connect Repository:**
   - Push your code to GitHub
   - Connect your repository to Render.com

2. **Deploy Services:**
   - **Database:** Create PostgreSQL database
   - **Backend:** Deploy web service from `render.yaml`
   - **Frontend:** Deploy static site from `render.yaml`

3. **Configure Environment Variables:**
   ```bash
   # Database
   DB_PASSWORD=your-secure-password

   # JWT Authentication
   JWT_SECRET_KEY=your-jwt-secret-key

   # Application Settings
   FLASK_ENV=production
   DEBUG=false
   ```

### Manual Deployment Steps

#### Step 1: Database Setup

1. **Create PostgreSQL Database:**
   - Go to Render Dashboard → New → PostgreSQL
   - Name: `supply-chain-db`
   - Database: `supply_chain`
   - User: `supply_user`

2. **Note the connection details** (will be used in environment variables)

#### Step 2: Backend Deployment

1. **Create Web Service:**
   - Go to Render Dashboard → New → Web Service
   - Name: `supply-chain-backend`
   - Runtime: `Python 3`
   - Build Command: `pip install -r ml-service/requirements.txt`
   - Start Command: `gunicorn --bind 0.0.0.0:$PORT --workers 4 --worker-class gfg app:app`

2. **Environment Variables:**
   ```bash
   DATABASE_URL=postgresql://supply_user:password@host:5432/supply_chain
   FLASK_ENV=production
   JWT_SECRET_KEY=your-secret-key
   DEBUG=false
   ```

#### Step 3: Frontend Deployment

1. **Create Static Site:**
   - Go to Render Dashboard → New → Static Site
   - Name: `supply-chain-frontend`
   - Build Command: `cd frontend && npm install && npm run build`
   - Publish Directory: `frontend/build`

2. **Environment Variables:**
   ```bash
   REACT_APP_API_URL=https://supply-chain-backend.onrender.com
   NODE_ENV=production
   ```

### Environment Variables Reference

| Variable | Description | Example |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection string | `postgresql://user:pass@host:5432/db` |
| `JWT_SECRET_KEY` | Secret key for JWT tokens | `your-secret-key` |
| `FLASK_ENV` | Flask environment | `production` |
| `DEBUG` | Debug mode | `false` |
| `REACT_APP_API_URL` | Backend API URL for frontend | `https://your-backend.onrender.com` |

### Cost Estimation

**Render.com Pricing (Approximate):**
- **PostgreSQL:** $7/month (512MB RAM)
- **Backend Service:** $7/month (512MB RAM)
- **Static Site:** Free
- **Total: $14/month** for basic setup

## 📋 Prerequisites (AWS)

- **AWS Account** with appropriate permissions
- **Terraform** installed (v1.0+)
- **AWS CLI** configured with credentials
- **Docker** and **Docker Compose** installed
- **Git** for version control

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Route 53      │    │   CloudFront     │    │   S3 Bucket     │
│   (DNS)         │────│   (CDN)          │────│   (Frontend)    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                        │
         │                        │
┌─────────────────┐    ┌──────────────────┐
│ Application     │    │   ECS Cluster    │
│ Load Balancer   │────│   (Backend)      │
│   (ALB)         │    │   (Auto-scaling) │
└─────────────────┘    └──────────────────┘
         │
┌─────────────────┐
│   RDS           │
│ PostgreSQL      │
│   (Multi-AZ)    │
└─────────────────┘
```

## 🚀 Quick Deployment

### Option 1: Automated Deployment (Recommended)

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd supply-chain-optimization
   ```

2. **Configure deployment variables:**
   ```bash
   cp terraform/terraform.tfvars.example terraform/terraform.tfvars
   # Edit terraform.tfvars with your values
   ```

3. **Run the deployment script:**
   ```bash
   chmod +x aws-deployment.sh
   ./aws-deployment.sh
   ```

### Option 2: Manual Deployment

#### Step 1: Set Up Infrastructure

1. **Initialize Terraform:**
   ```bash
   cd terraform
   terraform init
   ```

2. **Review and deploy infrastructure:**
   ```bash
   terraform plan
   terraform apply
   ```

3. **Note the outputs:**
   ```bash
   terraform output
   ```

#### Step 2: Build and Push Docker Images

1. **Build backend image:**
   ```bash
   docker build -f Dockerfile.backend -t supply-chain-backend:latest .
   ```

2. **Build frontend image:**
   ```bash
   docker build -f Dockerfile.frontend -t supply-chain-frontend:latest .
   ```

3. **Push to AWS ECR:**
   ```bash
   # Login to ECR
   aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com

   # Tag and push
   docker tag supply-chain-backend:latest YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/supply-chain-backend:latest
   docker tag supply-chain-frontend:latest YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/supply-chain-frontend:latest

   docker push YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/supply-chain-backend:latest
   docker push YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/supply-chain-frontend:latest
   ```

#### Step 3: Deploy to ECS

1. **Update ECS service:**
   ```bash
   aws ecs update-service --cluster supply-chain-cluster --service supply-chain-backend-service --force-new-deployment
   ```

2. **Wait for deployment:**
   ```bash
   aws ecs wait services-stable --cluster supply-chain-cluster --services supply-chain-backend-service
   ```

#### Step 4: Upload Frontend to S3

1. **Build frontend:**
   ```bash
   cd frontend
   npm run build
   cd ..
   ```

2. **Upload to S3:**
   ```bash
   aws s3 sync frontend/build s3://YOUR_FRONTEND_BUCKET --delete
   ```

3. **Invalidate CloudFront cache:**
   ```bash
   aws cloudfront create-invalidation --distribution-id YOUR_DISTRIBUTION_ID --paths "/*"
   ```

## ⚙️ Configuration

### Environment Variables

Create a `terraform.tfvars` file:

```hcl
aws_region = "us-east-1"
db_password = "your-secure-password"
domain_name = "your-domain.com"
backend_image = "YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/supply-chain-backend"
```

### Database Setup

The deployment creates a PostgreSQL RDS instance. Update your backend configuration:

```python
# In ml-service/config.py
DATABASE_URL = "postgresql://supply_user:your-password@your-db-endpoint:5432/supply_chain"
```

## 🔍 Monitoring and Logging

### CloudWatch Dashboards

- **ECS Service Metrics:** CPU, Memory, Task Count
- **Load Balancer Metrics:** Request Count, Response Time, Error Rate
- **Database Metrics:** Connections, CPU, Storage

### Logs

- **ECS Logs:** Available in CloudWatch Logs
- **Load Balancer Logs:** Enable access logs in S3
- **Application Logs:** Configured in container definitions

## 🔒 Security

### SSL/TLS Setup

1. **Request SSL Certificate:**
   ```bash
   aws acm request-certificate --domain-name your-domain.com --validation-method DNS
   ```

2. **Update Load Balancer Listener:**
   ```bash
   aws elbv2 create-listener --load-balancer-arn YOUR_ALB_ARN --protocol HTTPS --port 443 --ssl-policy ELBSecurityPolicy-2016-08 --certificates CertificateArn=YOUR_CERT_ARN --default-actions Type=forward,TargetGroupArn=YOUR_TARGET_GROUP_ARN
   ```

### Security Groups

- **ALB Security Group:** Allows HTTP/HTTPS from anywhere
- **ECS Security Group:** Allows traffic from ALB only
- **Database Security Group:** Allows traffic from ECS only

## 📊 Scaling

### Auto-Scaling Policies

- **CPU-based scaling:** Scales when CPU > 70%
- **Memory-based scaling:** Scales when Memory > 80%
- **Custom metrics:** Scale based on request rate or queue length

### Manual Scaling

```bash
aws ecs update-service --cluster supply-chain-cluster --service supply-chain-backend-service --desired-count 4
```

## 🔄 CI/CD Pipeline

The included GitHub Actions workflow automatically:

- Runs tests on pull requests
- Builds and pushes Docker images
- Deploys to ECS on main branch pushes
- Invalidates CloudFront cache

## 🛠️ Troubleshooting

### Common Issues

1. **ECS Tasks Failing:**
   - Check CloudWatch logs
   - Verify environment variables
   - Ensure security groups allow traffic

2. **Database Connection Issues:**
   - Verify RDS security group
   - Check database credentials
   - Ensure VPC configuration is correct

3. **Load Balancer Health Checks Failing:**
   - Verify target group health check path
   - Check ECS task health
   - Review security group rules

### Useful Commands

```bash
# Check ECS service status
aws ecs describe-services --cluster supply-chain-cluster --services supply-chain-backend-service

# View logs
aws logs tail /ecs/supply-chain-backend --follow

# Check load balancer health
aws elbv2 describe-target-health --target-group-arn YOUR_TARGET_GROUP_ARN

# Database connection test
aws rds describe-db-instances --db-instance-identifier supply-chain-db
```

## 📞 Support

For issues or questions:

1. Check CloudWatch logs and metrics
2. Review Terraform state and outputs
3. Verify AWS service quotas and limits
4. Contact AWS support if needed

## 🎯 Production Checklist

- [ ] SSL certificate installed and configured
- [ ] Domain name pointing to CloudFront distribution
- [ ] Database backups configured
- [ ] Monitoring and alerting set up
- [ ] Security groups reviewed and tightened
- [ ] Auto-scaling policies tested
- [ ] CI/CD pipeline working
- [ ] Load testing completed
- [ ] Documentation updated

## 💰 Cost Estimation

### Monthly Costs (Approximate)

- **ECS Fargate:** $20-50 (depending on usage)
- **RDS PostgreSQL:** $15-30 (t3.micro)
- **Load Balancer:** $20-30
- **CloudFront:** $1-5 (depending on traffic)
- **S3 Storage:** $0.50-2
- **CloudWatch:** $5-10

**Total: $60-130/month** for basic setup

---

🎉 **Congratulations!** Your Supply Chain Optimization System is now deployed with enterprise-grade scalability and reliability on AWS!
