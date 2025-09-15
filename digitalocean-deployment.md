# DigitalOcean Deployment Guide with VS Code Remote SSH

## Prerequisites
- DigitalOcean account
- VS Code with Remote-SSH extension installed
- Docker Desktop (for building images)

## Step 1: Create a DigitalOcean Droplet

1. Log into DigitalOcean
2. Create a new Droplet:
   - **Image**: Ubuntu 22.04 LTS x64
   - **Plan**: Basic - $6/month (1GB RAM, 1 CPU)
   - **Region**: Choose closest to you
   - **Authentication**: SSH Key (recommended) or Password
   - **Hostname**: `ppg-health-app`

3. Note your droplet's IP address (e.g., `YOUR_DROPLET_IP`)

## Step 2: Configure SSH Access

Add to your `~/.ssh/config`:
```
Host do-ppg
    HostName YOUR_DROPLET_IP
    User root
    Port 22
    IdentityFile ~/.ssh/id_rsa
```

Test connection:
```bash
ssh do-ppg
```

## Step 3: Connect VS Code to Droplet

1. Open VS Code
2. Press `Cmd+Shift+P` (Mac) or `Ctrl+Shift+P` (Windows/Linux)
3. Type "Remote-SSH: Connect to Host"
4. Select `do-ppg`
5. VS Code will open a new window connected to your droplet

## Step 4: Initial Server Setup (Run in VS Code Terminal)

Once connected via VS Code Remote SSH, run:

```bash
# Update system
apt update && apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh
rm get-docker.sh

# Install essential tools
apt install -y git nginx certbot python3-certbot-nginx

# Configure firewall
ufw allow 22/tcp
ufw allow 80/tcp
ufw allow 443/tcp
ufw allow 5000/tcp
ufw --force enable
```

## Step 5: Deploy Application

### Option A: Build on Server (Slower but simpler)
```bash
# Clone repository
git clone https://github.com/heyahmedsherif/bloodpressurepredictor.git
cd bloodpressurepredictor
git checkout camera-testing

# Build Docker image
docker build -f Dockerfile.flask -t ppg-app .

# Run container
docker run -d \
  --name ppg-health \
  --restart unless-stopped \
  -p 5000:5000 \
  ppg-app
```

### Option B: Push Pre-built Image (Faster)
On your local machine:
```bash
# Build for x64
docker buildx build --platform linux/amd64 -f Dockerfile.flask -t ppg-app .

# Tag for Docker Hub
docker tag ppg-app YOUR_DOCKERHUB_USERNAME/ppg-app:latest

# Push to Docker Hub
docker push YOUR_DOCKERHUB_USERNAME/ppg-app:latest
```

On the server (via VS Code):
```bash
# Pull and run
docker pull YOUR_DOCKERHUB_USERNAME/ppg-app:latest
docker run -d \
  --name ppg-health \
  --restart unless-stopped \
  -p 5000:5000 \
  YOUR_DOCKERHUB_USERNAME/ppg-app:latest
```

## Step 6: Configure Nginx Reverse Proxy

Create `/etc/nginx/sites-available/ppg-app`:
```nginx
server {
    listen 80;
    server_name YOUR_DROPLET_IP;
    
    client_max_body_size 100M;
    
    location / {
        proxy_pass http://localhost:5000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 180s;
        proxy_connect_timeout 180s;
        proxy_send_timeout 180s;
    }
}
```

Enable the site:
```bash
ln -s /etc/nginx/sites-available/ppg-app /etc/nginx/sites-enabled/
nginx -t
systemctl restart nginx
```

## Step 7: Access Your App

1. Open browser: `http://YOUR_DROPLET_IP`
2. The app should be running!

## Step 8: SSL Certificate (Optional - requires domain)

If you have a domain pointing to your droplet:
```bash
certbot --nginx -d yourdomain.com
```

## VS Code Remote Development Tips

1. **Install extensions on remote**: Docker, Python, GitLens
2. **Port forwarding**: VS Code auto-forwards ports
3. **Terminal**: Use integrated terminal for all commands
4. **File editing**: Edit files directly on server
5. **Debugging**: Can debug Docker containers remotely

## Monitoring Commands

```bash
# Check container status
docker ps -a

# View logs
docker logs ppg-health -f

# Restart container
docker restart ppg-health

# Check system resources
htop

# Check nginx logs
tail -f /var/log/nginx/access.log
tail -f /var/log/nginx/error.log
```

## Troubleshooting

### If container won't start:
```bash
docker logs ppg-health
docker rm ppg-health
# Then re-run docker run command
```

### If nginx isn't working:
```bash
systemctl status nginx
nginx -t
systemctl restart nginx
```

### If out of memory:
Consider upgrading to $12/month droplet (2GB RAM)

## Cost Estimate
- Basic Droplet (1GB): $6/month
- With backups: +$1.20/month
- Total: ~$7.20/month

## Next Steps
1. Set up automated backups
2. Configure monitoring (e.g., UptimeRobot)
3. Add a domain name
4. Enable SSL certificate