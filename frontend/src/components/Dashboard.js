import React, { useState, useEffect, useRef } from 'react';
import * as THREE from 'three';
import Lenis from '@studio-freight/lenis';
import ApiService from '../api';
import WebSocketService from '../services/websocket';
import './Dashboard.css';

function Dashboard({ token }) {
  const [metrics, setMetrics] = useState({
    totalInventory: 0,
    activeOrders: 0,
    delayedShipments: 0,
    totalRevenue: 0,
    demandForecast: 0,
    supplierPerformance: 0,
    systemHealth: 95,
    lastUpdate: new Date().toLocaleTimeString(),
  });

  const [etaData, setEtaData] = useState([]);

  const [alerts, setAlerts] = useState([]);
  const [isConnected, setIsConnected] = useState(false);
  const [showDrillDown, setShowDrillDown] = useState(null);
  const [selectedTimeRange, setSelectedTimeRange] = useState('24h');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [drillDownData, setDrillDownData] = useState(null);
  const [lastFetch, setLastFetch] = useState(0);
  const [currentTime, setCurrentTime] = useState(new Date());
  const [animatedMetrics, setAnimatedMetrics] = useState({
    totalInventory: 0,
    activeOrders: 0,
    delayedShipments: 0,
    totalRevenue: 0,
    demandForecast: 0,
    supplierPerformance: 0,
    systemHealth: 95,
  });

  const threeContainerRef = useRef(null);
  const rendererRef = useRef(null);

  // Fetch dashboard data
  const fetchDashboardData = async () => {
    const now = Date.now();
    if (now - lastFetch < 2000) { // 2 second cooldown to avoid 429
      console.log('Rate limit: Wait before next fetch');
      setError('Please wait a moment before refreshing.');
      return;
    }
    setLastFetch(now);
    setLoading(true);
    setError(null);
    try {
      const data = await ApiService.getDashboardMetrics(token);
      setMetrics({ ...data, lastUpdate: new Date().toLocaleTimeString() });
    } catch (err) {
      if (err.response?.status === 401) {
        // Token expired, logout user
        localStorage.removeItem('token');
        localStorage.removeItem('user');
        window.location.reload(); // Force reload to show login
      } else if (err.response?.status === 429) {
        setError('Too many requests. Please wait and try again.');
      } else {
        setError('Failed to fetch dashboard data');
        console.error('Error fetching dashboard data:', err.response?.data?.error || err.message);
      }
    } finally {
      setLoading(false);
    }
  };

  // Fetch ETA data
  const fetchEtaData = async () => {
    try {
      const data = await ApiService.getEtaEstimates(token);
      setEtaData(data.etaData || []);
    } catch (err) {
      if (err.response?.status === 401) {
        // Token expired, logout user
        localStorage.removeItem('token');
        localStorage.removeItem('user');
        window.location.reload(); // Force reload to show login
      } else {
        console.error('Error fetching ETA data:', err.response?.data?.error || err.message);
      }
    }
  };

  useEffect(() => {
    // Initialize Lenis for smooth scrolling
    const lenis = new Lenis({
      duration: 1.2,
      easing: (t) => Math.min(1, 1.001 - Math.pow(2, -10 * t)),
      direction: 'vertical',
      gestureDirection: 'vertical',
      smooth: true,
      mouseMultiplier: 1,
      smoothTouch: false,
      touchMultiplier: 2,
      infinite: false,
    });

    function raf(time) {
      lenis.raf(time);
      requestAnimationFrame(raf);
    }

    requestAnimationFrame(raf);

    // Update current time every second
    const timer = setInterval(() => setCurrentTime(new Date()), 1000);

    // Temporarily disable WebSocket to prevent login redirect issue
    // TODO: Fix WebSocket backend error and re-enable
    /*
    WebSocketService.connect(token);

    const handleConnect = () => setIsConnected(true);
    const handleDisconnect = () => setIsConnected(false);
    const handleConnectError = (err) => {
      setIsConnected(false);
      // Only logout on actual authentication failures, not other WebSocket errors
      if (err.message.includes('Authentication failed') || err.message.includes('expired') || err.message.includes('Invalid token')) {
        // Token expired, logout user
        localStorage.removeItem('token');
        localStorage.removeItem('user');
        window.location.reload(); // Force reload to show login
      }
    };

    WebSocketService.on('connect', handleConnect);
    WebSocketService.on('disconnect', handleDisconnect);
    WebSocketService.on('connect_error', handleConnectError);

    // Subscriptions
    WebSocketService.subscribe('dashboard_metrics', (data) => {
      setMetrics({ ...data, lastUpdate: new Date().toLocaleTimeString() });
    });
    WebSocketService.subscribe('inventory_update', (data) => {
      setMetrics((prev) => ({ ...prev, totalInventory: data.total_inventory }));
    });
    WebSocketService.subscribe('order_update', (data) => {
      setMetrics((prev) => ({ ...prev, activeOrders: data.active_orders }));
    });
    WebSocketService.subscribe('alert', (data) => {
      setAlerts((prev) => [data, ...prev.slice(0, 9)]);
    });
    */

    // Initial fetch
    fetchDashboardData();
    fetchEtaData();

    return () => {
      lenis.destroy();
      clearInterval(timer);
      /*
      WebSocketService.unsubscribe('dashboard_metrics');
      WebSocketService.unsubscribe('inventory_update');
      WebSocketService.unsubscribe('order_update');
      WebSocketService.unsubscribe('alert');
      WebSocketService.off('connect', handleConnect);
      WebSocketService.off('disconnect', handleDisconnect);
      WebSocketService.off('connect_error', handleConnectError);
      WebSocketService.disconnect();
      */
    };
  }, [token]);

  // Animate metrics when they change
  useEffect(() => {
    const animateMetrics = () => {
      const keys = Object.keys(metrics);
      keys.forEach(key => {
        if (typeof metrics[key] === 'number' && key !== 'lastUpdate') {
          const start = animatedMetrics[key] || 0;
          const end = metrics[key];
          const duration = 2000; // 2 seconds
          const startTime = Date.now();

          const animate = () => {
            const elapsed = Date.now() - startTime;
            const progress = Math.min(elapsed / duration, 1);
            const easeOut = 1 - Math.pow(1 - progress, 3); // Ease out cubic
            const current = start + (end - start) * easeOut;

            setAnimatedMetrics(prev => ({ ...prev, [key]: Math.round(current) }));

            if (progress < 1) {
              requestAnimationFrame(animate);
            }
          };

          requestAnimationFrame(animate);
        }
      });
    };

    animateMetrics();
  }, [metrics]);

  // Three.js setup for Supply Chain Visualization
  useEffect(() => {
    const container = threeContainerRef.current;
    if (!container) return;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0a0e1a); // Dark background matching theme

    const camera = new THREE.PerspectiveCamera(75, container.clientWidth / container.clientHeight, 0.1, 1000);
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    container.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    // Reduced Star Field for Premium Feel
    const starGeometry = new THREE.BufferGeometry();
    const starCount = 500; // Reduced density
    const starPositions = new Float32Array(starCount * 3);
    for (let i = 0; i < starCount * 3; i++) {
      starPositions[i] = (Math.random() - 0.5) * 2000;
    }
    starGeometry.setAttribute('position', new THREE.BufferAttribute(starPositions, 3));
    const starMaterial = new THREE.PointsMaterial({
      color: 0x00ffff,
      size: 1.5,
      sizeAttenuation: false,
      transparent: true,
      opacity: 0.6
    });
    const stars = new THREE.Points(starGeometry, starMaterial);
    scene.add(stars);

    // Supply Chain Nodes Data
    const nodesData = [
      { name: 'Supplier 1', position: [-4, 2, 0], color: 0x10b981, size: 0.8 },
      { name: 'Supplier 2', position: [-4, -2, 0], color: 0xf59e0b, size: 0.6 },
      { name: 'Warehouse', position: [0, 0, 0], color: 0x3b82f6, size: 1.2 },
      { name: 'DC 1', position: [4, 2, 0], color: 0x8b5cf6, size: 0.9 },
      { name: 'DC 2', position: [4, -2, 0], color: 0x06b6d4, size: 0.9 },
      { name: 'Customer 1', position: [8, 1, 0], color: 0xef4444, size: 0.7 },
      { name: 'Customer 2', position: [8, -1, 0], color: 0x10b981, size: 0.7 },
    ];

    // Lighting
    const ambientLight = new THREE.AmbientLight(0x404040, 0.4);
    scene.add(ambientLight);
    const directionalLight = new THREE.DirectionalLight(0x8b5cf6, 1.5);
    directionalLight.position.set(5, 5, 5);
    directionalLight.castShadow = true;
    scene.add(directionalLight);

    nodesData.forEach((nodeData) => {
      const pointLight = new THREE.PointLight(nodeData.color, 1, 10);
      pointLight.position.set(...nodeData.position);
      scene.add(pointLight);
    });

    // Colored Edges with Status
    const edges = [
      { from: 0, to: 2, status: 'on-time', color: 0x10b981 }, // Green
      { from: 1, to: 2, status: 'risky', color: 0xf59e0b }, // Yellow
      { from: 2, to: 3, status: 'on-time', color: 0x10b981 },
      { from: 2, to: 4, status: 'delayed', color: 0xef4444 }, // Red
      { from: 3, to: 5, status: 'risky', color: 0xf59e0b },
      { from: 3, to: 6, status: 'on-time', color: 0x10b981 },
      { from: 4, to: 5, status: 'delayed', color: 0xef4444 },
      { from: 4, to: 6, status: 'on-time', color: 0x10b981 },
    ];

    // Create nodes
    const nodes = [];
    nodesData.forEach((nodeData) => {
      const geometry = new THREE.SphereGeometry(nodeData.size, 32, 32);
      const material = new THREE.MeshPhongMaterial({ 
        color: nodeData.color,
        emissive: new THREE.Color(nodeData.color).multiplyScalar(0.2),
        shininess: 100,
        specular: 0x111111
      });
      const node = new THREE.Mesh(geometry, material);
      node.position.set(...nodeData.position);
      node.castShadow = true;
      node.receiveShadow = true;
      scene.add(node);
      nodes.push(node);
    });

    // Create colored edges
    const lines = [];
    edges.forEach((edge) => {
      const points = [];
      points.push(new THREE.Vector3(...nodesData[edge.from].position));
      points.push(new THREE.Vector3(...nodesData[edge.to].position));
      const geometry = new THREE.BufferGeometry().setFromPoints(points);
      const lineMaterial = new THREE.LineBasicMaterial({ 
        color: edge.color, 
        transparent: true, 
        opacity: 0.8,
        linewidth: 4
      });
      const line = new THREE.Line(geometry, lineMaterial);
      scene.add(line);
      lines.push(line);
    });

    // Cars (simple box geometries with colors)
    const cars = [];
    const carPaths = edges.map((edge, index) => ({
      path: [
        new THREE.Vector3(...nodesData[edge.from].position),
        new THREE.Vector3(...nodesData[edge.to].position)
      ],
      status: edge.status,
      color: edge.color,
      progress: 0
    }));

    carPaths.forEach((path, index) => {
      const carGeometry = new THREE.BoxGeometry(0.2, 0.1, 0.1);
      const carMaterial = new THREE.MeshPhongMaterial({ color: path.color });
      const car = new THREE.Mesh(carGeometry, carMaterial);
      car.position.copy(path.path[0]);
      scene.add(car);
      cars.push({ mesh: car, path, index });
    });

    // Reduced particles
    const particlesGeometry = new THREE.BufferGeometry();
    const particlesCount = 50;
    const posArray = new Float32Array(particlesCount * 3);
    for (let i = 0; i < particlesCount * 3; i++) {
      posArray[i] = (Math.random() - 0.5) * 20;
    }
    particlesGeometry.setAttribute('position', new THREE.BufferAttribute(posArray, 3));
    const particlesMaterial = new THREE.PointsMaterial({
      color: 0x3b82f6,
      size: 0.03,
      transparent: true,
      opacity: 0.7
    });
    const particlesMesh = new THREE.Points(particlesGeometry, particlesMaterial);
    scene.add(particlesMesh);

    scene.fog = new THREE.Fog(0x0a0e1a, 10, 50);

    camera.position.set(0, 0, 12);
    camera.lookAt(0, 0, 0);

    let time = 0;
    const animate = () => {
      requestAnimationFrame(animate);
      time += 0.01;

      // Gentle rotation
      scene.rotation.y += 0.001;

      // Node pulsing (subtle)
      nodes.forEach((node, index) => {
        const pulse = 1 + Math.sin(time + index) * 0.1;
        node.scale.setScalar(pulse);
        node.material.emissiveIntensity = 0.2 + Math.sin(time * 1.5 + index) * 0.1;
        node.rotation.y += 0.003;
      });

      // Animate cars along paths
      cars.forEach((carObj) => {
        carObj.path.progress += 0.005; // Speed
        if (carObj.path.progress > 1) {
          carObj.path.progress = 0;
          carObj.mesh.position.copy(carObj.path.path[0]);
        }
        const currentPos = new THREE.Vector3().lerpVectors(
          carObj.path.path[0],
          carObj.path.path[1],
          carObj.path.progress
        );
        carObj.mesh.position.copy(currentPos);
        carObj.mesh.rotation.y += 0.05; // Rotate car
      });

      // Stars subtle twinkle
      stars.rotation.y += 0.0002;
      stars.material.opacity = 0.6 + Math.sin(time * 0.3) * 0.1;

      // Particles flow
      const positions = particlesMesh.geometry.attributes.position.array;
      for (let i = 0; i < positions.length; i += 3) {
        positions[i + 1] += Math.sin(time + positions[i]) * 0.01;
        if (positions[i + 1] > 5) positions[i + 1] = -5;
        positions[i + 2] += Math.cos(time * 0.3) * 0.005;
      }
      particlesMesh.geometry.attributes.position.needsUpdate = true;
      particlesMesh.material.opacity = 0.7 + Math.sin(time) * 0.05;

      renderer.render(scene, camera);
    };
    animate();

    const handleResize = () => {
      camera.aspect = container.clientWidth / container.clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(container.clientWidth, container.clientHeight);
    };
    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      if (renderer.domElement) {
        container.removeChild(renderer.domElement);
      }
      nodes.forEach(node => {
        scene.remove(node);
        node.geometry.dispose();
        node.material.dispose();
      });
      cars.forEach(carObj => {
        scene.remove(carObj.mesh);
        carObj.mesh.geometry.dispose();
        carObj.mesh.material.dispose();
      });
      renderer.dispose();
    };
  }, []);

  const handleMetricClick = (metricType) => {
    setShowDrillDown(metricType);
    setDrillDownData(null);
  };

  const closeDrillDown = () => {
    setShowDrillDown(null);
    setDrillDownData(null);
  };

  const handleModalClick = (e) => {
    if (e.target === e.currentTarget) {
      closeDrillDown();
    }
  };

  const getHealthColor = (health) => {
    if (health >= 90) return 'excellent';
    if (health >= 70) return 'good';
    if (health >= 50) return 'warning';
    return 'critical';
  };

  const getHealthIcon = (health) => {
    if (health >= 90) return '🟢';
    if (health >= 70) return '🟡';
    if (health >= 50) return '🟠';
    return '🔴';
  };

  return (
    <div className="dashboard premium-black">
      {/* Connection Status */}
      <div className={`connection-status ${isConnected ? 'connected' : 'disconnected'}`}>
        <span className="status-indicator"></span>
        {isConnected ? 'Live Data' : 'Connecting...'}
      </div>

      {/* Header */}
      <div className="dashboard-header">
        <div className="header-left">
          <h2 className="premium-title">🚀 Quantum Supply Chain Command Center</h2>
          <p className="subtext">AI-Powered Real-Time Operations</p>
          <p className="last-update">Last updated: {metrics.lastUpdate}</p>
          <p className="current-time">Current Time: {currentTime.toLocaleTimeString()}</p>
          {loading && <p className="loading-text">Loading data...</p>}
          {error && <p className="error-text">{error}</p>}
        </div>
        <div className="header-actions">
          <select
            value={selectedTimeRange}
            onChange={(e) => setSelectedTimeRange(e.target.value)}
            className="time-range-selector"
          >
            <option value="1h">Last Hour</option>
            <option value="24h">Last 24 Hours</option>
            <option value="7d">Last 7 Days</option>
            <option value="30d">Last 30 Days</option>
          </select>
          <button onClick={fetchDashboardData} className="refresh-button" disabled={loading}>
            🔄 Refresh
          </button>
        </div>
      </div>

      {/* 3D Supply Chain Visualization */}
      <div className="three-section">
        <h3>3D Supply Chain Visualization</h3>
        <div ref={threeContainerRef} className="three-container"></div>
      </div>

      {/* System Health */}
      <div className="system-health">
        <div className="health-card circular-health">
          <div className="health-icon">{getHealthIcon(metrics.systemHealth)}</div>
          <div className="health-info">
            <h3>🛡️ Cyber Health Index</h3>
            <p className={`health-score ${getHealthColor(metrics.systemHealth)}`}>
              {metrics.systemHealth}%
            </p>
            <svg className="circular-progress" viewBox="0 0 36 36">
              <path
                className="progress-bg"
                d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
              />
              <path
                className="progress-fill"
                strokeDasharray={`${metrics.systemHealth * 1.11}, 100`}
                d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
              />
            </svg>
          </div>
        </div>
      </div>

      {/* KPI Metrics */}
      <div className="metrics-grid">
        <div className="metric-card clickable inventory-card" onClick={() => handleMetricClick('inventory')}>
          <div className="metric-icon">📦</div>
          <h3>Quantum Inventory Flow</h3>
          <p className="metric-value">${animatedMetrics.totalInventory.toLocaleString()}</p>
          <div className="metric-3d-chart">
            <div className="metric-bar"></div>
          </div>
          <span className="trend positive">+2%</span>
        </div>

        <div className="metric-card clickable orders-card" onClick={() => handleMetricClick('orders')}>
          <div className="metric-icon">📋</div>
          <h3>Order Stream Intelligence</h3>
          <p className="metric-value">{animatedMetrics.activeOrders}</p>
          <span className="trend negative">-1%</span>
        </div>

        <div className="metric-card clickable shipments-card" onClick={() => handleMetricClick('shipments')}>
          <div className="metric-icon">🚚</div>
          <h3>Shipment Velocity Tracker</h3>
          <p className="metric-value">{animatedMetrics.delayedShipments}</p>
          <span className="trend positive">+5%</span>
        </div>

        <div className="metric-card clickable revenue-card" onClick={() => handleMetricClick('revenue')}>
          <div className="metric-icon">💰</div>
          <h3>Revenue Pulse Engine</h3>
          <p className="metric-value">${animatedMetrics.totalRevenue.toLocaleString()}</p>
          <span className="trend positive">+10%</span>
        </div>

        <div className="metric-card forecast-card" onClick={() => handleMetricClick('forecast')}>
          <div className="metric-icon">🎯</div>
          <h3>Predictive Demand Radar</h3>
          <p className="metric-value">{animatedMetrics.demandForecast}%</p>
          <span className="trend positive">+3%</span>
        </div>

        <div className="metric-card supplier-card" onClick={() => handleMetricClick('supplier')}>
          <div className="metric-icon">⭐</div>
          <h3>Supplier Trust Index</h3>
          <p className="metric-value">{animatedMetrics.supplierPerformance}%</p>
          <span className="trend positive">+1%</span>
        </div>
      </div>

      {/* ETA + Timing Section */}
      <div className="eta-section">
        <h3> Estimated Time of Arrival+ Timing</h3>
        <div className="eta-rings">
          {etaData.map((delivery) => (
            <div key={delivery.id} className={`eta-ring ${delivery.status}`}>
              <div className="ring-progress" style={{ '--progress': `${delivery.progress}%` }}></div>
              <div className="car-icon">🚗</div>
              <p className="eta-text">ETA: {delivery.eta}</p>
            </div>
          ))}
        </div>
      </div>

      {/* Alerts */}
      <div className="alerts-section">
        <h3>⚡ Live Event Stream</h3>
        <div className="alerts-list">
          {alerts.map((alert, index) => (
            <div key={index} className={`alert-item ${alert.priority || 'info'}`}>
              <span className="alert-time">{alert.timestamp}</span>
              <p className="alert-message">{alert.message}</p>
            </div>
          ))}
        </div>
      </div>

      {/* Drill-down Modal */}
      {showDrillDown && (
        <div className="drilldown-modal holographic" onClick={handleModalClick}>
          <div className="drilldown-content">
            <button className="close-button" onClick={closeDrillDown}>✖</button>
            <h3>🔍 Deep Data Dive: {showDrillDown}</h3>
            {drillDownData ? <pre>{JSON.stringify(drillDownData, null, 2)}</pre> :
              <p>No detailed data available.</p>}
          </div>
        </div>
      )}
    </div>
  );
}

export default Dashboard;
