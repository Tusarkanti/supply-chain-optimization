import React, { useState, useEffect } from 'react';
import ApiService from '../api';
import './Login.css';
import '../Starfield.css';

function Login({ onLogin }) {
  const [isLogin, setIsLogin] = useState(true);
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    password: '',
    confirmPassword: ''
  });
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [success, setSuccess] = useState('');

  // Load saved data on component mount
  useEffect(() => {
    const savedEmail = localStorage.getItem('supplychain_email');
    const savedName = localStorage.getItem('supplychain_name');

    if (savedEmail) {
      setFormData(prev => ({
        ...prev,
        email: savedEmail
      }));
    }

    if (savedName) {
      setFormData(prev => ({
        ...prev,
        name: savedName
      }));
    }
  }, []);

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    });
    // Clear errors when user starts typing
    if (error) setError('');
    if (success) setSuccess('');
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setSuccess('');
    setLoading(true);

    try {
      const endpoint = isLogin ? '/api/login' : '/api/register';
      const payload = isLogin
        ? { email: formData.email, password: formData.password }
        : { name: formData.name, email: formData.email, password: formData.password };

      // Validate login
      if (isLogin) {
        if (!formData.password.trim()) {
          setError('Password is required');
          setLoading(false);
          return;
        }
      }

      // Validate registration
      if (!isLogin) {
        if (formData.password !== formData.confirmPassword) {
          setError('Passwords do not match');
          setLoading(false);
          return;
        }
        if (formData.password.length < 6) {
          setError('Password must be at least 6 characters long');
          setLoading(false);
          return;
        }
      }

      console.log(`Sending ${isLogin ? 'login' : 'registration'} data:`, payload);

      const data = await (isLogin ? ApiService.login(payload) : ApiService.register(payload));

      // Save user data to localStorage for persistence
      localStorage.setItem('supplychain_email', formData.email);
      if (formData.name) {
        localStorage.setItem('supplychain_name', formData.name);
      }

      if (isLogin) {
        onLogin(data.access_token, data.user);
      } else {
        setSuccess('Account created successfully! You can now sign in.');
        setIsLogin(true);
        setFormData({
          name: '',
          email: '',
          password: '',
          confirmPassword: ''
        });
      }
    } catch (err) {
      setError(err.error || 'Network error. Please check if the server is running.');
      console.error('Network error:', err);
    } finally {
      setLoading(false);
    }
  };

  const toggleMode = () => {
    setIsLogin(!isLogin);
    setError('');
    setSuccess('');
    setFormData({
      name: '',
      email: '',
      password: '',
      confirmPassword: ''
    });
  };

  return (
    <div className="login-container">
      <div className="login-background">
        <div className="login-card">
          <div className="login-header">
            <div className="logo-section">
              <div className="logo-icon">
                <svg viewBox="0 0 24 24" fill="currentColor">
                  <path d="M12 2L2 7v10c0 5.55 3.84 9.74 9 11 5.16-1.26 9-5.45 9-11V7l-10-5z"/>
                </svg>
              </div>
              <h1 className="app-title">SupplyChain Pro</h1>
              <p className="app-subtitle">Advanced Optimization Platform</p>
            </div>
          </div>

          <div className="login-body">
            <div className="form-header">
              <h2 className="form-title">
                {isLogin ? 'Welcome Back' : 'Create Account'}
              </h2>
              <p className="form-subtitle">
                {isLogin
                  ? 'Sign in to access your dashboard'
                  : 'Join thousands of businesses optimizing their supply chain'
                }
              </p>
            </div>

            <form onSubmit={handleSubmit} className="login-form">
              {!isLogin && (
                <div className="form-group">
                  <label htmlFor="name">Full Name</label>
                  <div className="input-wrapper">
                    <input
                      type="text"
                      id="name"
                      name="name"
                      value={formData.name}
                      onChange={handleChange}
                      placeholder="Enter your full name"
                      required
                      disabled={loading}
                    />
                  </div>
                </div>
              )}

              <div className="form-group">
                <label htmlFor="email">Email Address</label>
                <div className="input-wrapper">
                  <input
                    type="email"
                    id="email"
                    name="email"
                    value={formData.email}
                    onChange={handleChange}
                    placeholder="Enter your email address"
                    required
                    disabled={loading}
                    autoComplete="email"
                  />
                </div>
              </div>

              <div className="form-group">
                <label htmlFor="password">Password</label>
                <div className="input-wrapper">
                  <input
                    type="password"
                    id="password"
                    name="password"
                    value={formData.password}
                    onChange={handleChange}
                    placeholder={isLogin ? "Enter your password" : "Create a secure password"}
                    required
                    disabled={loading}
                    autoComplete={isLogin ? "current-password" : "new-password"}
                  />
                </div>
              </div>

              {!isLogin && (
                <div className="form-group">
                  <label htmlFor="confirmPassword">Confirm Password</label>
                  <div className="input-wrapper">
                    <input
                      type="password"
                      id="confirmPassword"
                      name="confirmPassword"
                      value={formData.confirmPassword}
                      onChange={handleChange}
                      placeholder="Confirm your password"
                      required
                      disabled={loading}
                      autoComplete="new-password"
                    />
                  </div>
                </div>
              )}

              {error && <div className="error-message">{error}</div>}
              {success && <div className="success-message">{success}</div>}

              <button
                type="submit"
                className="login-button"
                disabled={loading}
              >
                {loading ? (
                  <span className="loading-state">
                    <span className="spinner"></span>
                    Processing...
                  </span>
                ) : (
                  isLogin ? 'Sign In' : 'Create Account'
                )}
              </button>
            </form>

            <div className="auth-toggle">
              <p>
                {isLogin ? "Don't have an account?" : "Already have an account?"}
                <button
                  type="button"
                  onClick={toggleMode}
                  className="toggle-button"
                  disabled={loading}
                >
                  {isLogin ? 'Create one' : 'Sign in'}
                </button>
              </p>
            </div>

            {isLogin && (
              <div className="demo-credentials">
                <h4>Demo Credentials</h4>
                <div className="credential-item">
                  <span className="label">Email:</span>
                  <code className="value">admin@supplychain.com</code>
                </div>
                <div className="credential-item">
                  <span className="label">Password:</span>
                  <code className="value">admin123</code>
                </div>
                <p className="demo-note">Alternative: demo@supplychain.com / demo123</p>
                <button
                  type="button"
                  className="demo-fill-button"
                  onClick={async () => {
                    const credentials = { email: 'admin@supplychain.com', password: 'admin123' };
                    setFormData(prev => ({
                      ...prev,
                      email: credentials.email,
                      password: credentials.password
                    }));
                    setLoading(true);
                    setError('');
                    setSuccess('');
                    try {
                      const data = await ApiService.login(credentials);
                      localStorage.setItem('supplychain_email', credentials.email);
                      onLogin(data.access_token, data.user);
                    } catch (err) {
                      setError(err.error || 'Network error. Please check if the server is running.');
                      console.error('Auto-submit error:', err);
                    } finally {
                      setLoading(false);
                    }
                  }}
                >
                  Fill Demo Credentials & Sign In
                </button>
                <button
                  type="button"
                  className="clear-data-button"
                  onClick={() => {
                    localStorage.removeItem('supplychain_email');
                    localStorage.removeItem('supplychain_name');
                    setFormData(prev => ({
                      ...prev,
                      email: '',
                      name: ''
                    }));
                  }}
                >
                  Clear Saved Data
                </button>
              </div>
            )}
          </div>
        </div>
      </div>

      <footer className="login-footer">
        <p>DEDICATED TO THE DEAREST FRD BUDDI 🥰</p>
      </footer>
    </div>
  );
}

export default Login;