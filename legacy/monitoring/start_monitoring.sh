#!/bin/bash

# 启动 LLM 监控系统
# 包括 Prometheus 和 Grafana

set -e

echo "=================================="
echo "🤖 启动 LLM 监控系统"
echo "=================================="

# 检查 Docker 和 Docker Compose
if ! command -v docker &> /dev/null; then
    echo "❌ Docker 未安装"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose 未安装"
    exit 1
fi

# 启动容器
echo ""
echo "📦 启动 Prometheus 和 Grafana..."
docker-compose up -d

echo ""
echo "⏳ 等待服务启动..."
sleep 3

# 检查服务
echo ""
echo "✅ 服务已启动！"
echo ""
echo "📊 访问地址："
echo "  • Prometheus: http://localhost:9090"
echo "  • Grafana:    http://localhost:3001"
echo "    - 用户名: admin"
echo "    - 密码: admin"
echo ""
echo "🔍 验证监控系统："
echo "  python verify_monitoring.py"
echo ""
echo "📚 文档："
echo "  • 快速入门: MONITORING_GUIDE.md"
echo "  • 集成说明: MONITORING_SETUP.md"
echo ""
