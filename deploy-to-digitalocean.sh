#!/bin/bash

# DigitalOcean Deployment Script for PPG Health App
# Run this script on your DigitalOcean droplet after connecting via VS Code Remote SSH

set -e  # Exit on error

echo "========================================="
echo "PPG Health App - DigitalOcean Deployment"
echo "========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

# Check if running as root
if [[ $EUID -ne 0 ]]; then
   print_error "This script must be run as root"
   exit 1
fi

# Step 1: Update System
print_status "Updating system packages..."
apt update && apt upgrade -y

# Step 2: Install Docker if not present
if ! command -v docker &> /dev/null; then
    print_status "Installing Docker..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sh get-docker.sh
    rm get-docker.sh
    systemctl enable docker
    systemctl start docker
else
    print_status "Docker already installed"
fi

# Step 3: Install additional tools
print_status "Installing additional tools..."
apt install -y git nginx certbot python3-certbot-nginx htop

# Step 4: Configure firewall
print_status "Configuring firewall..."
ufw allow 22/tcp
ufw allow 80/tcp
ufw allow 443/tcp
ufw allow 5000/tcp
echo "y" | ufw enable

# Step 5: Clone repository
if [ ! -d "/opt/ppg-app" ]; then
    print_status "Cloning repository..."
    cd /opt
    git clone https://github.com/heyahmedsherif/bloodpressurepredictor.git ppg-app
    cd ppg-app
    git checkout camera-testing
else
    print_status "Repository already exists, pulling latest changes..."
    cd /opt/ppg-app
    git pull origin camera-testing
fi

# Step 6: Stop and remove existing container if exists
if [ "$(docker ps -aq -f name=ppg-health)" ]; then
    print_warning "Stopping existing container..."
    docker stop ppg-health
    docker rm ppg-health
fi

# Step 7: Build Docker image
print_status "Building Docker image (this may take a few minutes)..."
docker build -f Dockerfile.flask -t ppg-app .

# Step 8: Run Docker container
print_status "Starting Docker container..."
docker run -d \
  --name ppg-health \
  --restart unless-stopped \
  -p 5000:5000 \
  --memory="1g" \
  --cpus="1.0" \
  ppg-app

# Step 9: Configure Nginx
print_status "Configuring Nginx reverse proxy..."

# Get server IP
SERVER_IP=$(curl -s ifconfig.me)

cat > /etc/nginx/sites-available/ppg-app << EOF
server {
    listen 80;
    server_name $SERVER_IP;
    
    client_max_body_size 100M;
    proxy_read_timeout 180s;
    proxy_connect_timeout 180s;
    proxy_send_timeout 180s;
    
    location / {
        proxy_pass http://localhost:5000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host \$host;
        proxy_cache_bypass \$http_upgrade;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
    
    # WebSocket support for real-time features
    location /socket.io {
        proxy_pass http://localhost:5000/socket.io;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
}
EOF

# Enable the site
ln -sf /etc/nginx/sites-available/ppg-app /etc/nginx/sites-enabled/
rm -f /etc/nginx/sites-enabled/default

# Test nginx configuration
nginx -t

# Restart nginx
systemctl restart nginx

# Step 10: Create update script
print_status "Creating update script..."
cat > /usr/local/bin/update-ppg-app << 'EOF'
#!/bin/bash
cd /opt/ppg-app
git pull origin camera-testing
docker stop ppg-health
docker rm ppg-health
docker build -f Dockerfile.flask -t ppg-app .
docker run -d \
  --name ppg-health \
  --restart unless-stopped \
  -p 5000:5000 \
  --memory="1g" \
  --cpus="1.0" \
  ppg-app
echo "App updated successfully!"
EOF

chmod +x /usr/local/bin/update-ppg-app

# Step 11: Create monitoring script
print_status "Creating monitoring script..."
cat > /usr/local/bin/check-ppg-app << 'EOF'
#!/bin/bash
echo "=== PPG Health App Status ==="
echo ""
echo "Docker Container:"
docker ps -a | grep ppg-health
echo ""
echo "Container Logs (last 20 lines):"
docker logs ppg-health --tail 20
echo ""
echo "System Resources:"
free -h
echo ""
echo "Disk Usage:"
df -h /
echo ""
echo "Nginx Status:"
systemctl status nginx --no-pager | head -n 10
EOF

chmod +x /usr/local/bin/check-ppg-app

# Step 12: Verify deployment
print_status "Verifying deployment..."
sleep 5  # Wait for container to fully start

if curl -s -o /dev/null -w "%{http_code}" http://localhost:5000 | grep -q "200\|302"; then
    print_status "Application is running successfully!"
else
    print_error "Application may not be running correctly. Check logs with: docker logs ppg-health"
fi

# Print summary
echo ""
echo "========================================="
echo "         DEPLOYMENT COMPLETE!"
echo "========================================="
echo ""
print_status "Your app is now accessible at:"
echo "  http://$SERVER_IP"
echo ""
print_status "Useful commands:"
echo "  check-ppg-app     - Check app status"
echo "  update-ppg-app    - Update app from git"
echo "  docker logs ppg-health -f    - View live logs"
echo "  docker restart ppg-health    - Restart container"
echo ""
print_warning "Note: The app needs HTTPS to access camera on most browsers."
print_warning "Consider adding a domain and SSL certificate."
echo ""
echo "========================================="