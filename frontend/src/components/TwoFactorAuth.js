import React, { useState } from 'react';
import './TwoFactorAuth.css';

function TwoFactorAuth({ onVerify, onCancel }) {
  const [code, setCode] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');

    try {
      const response = await fetch('http://localhost:5000/api/verify-2fa', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ code }),
      });

      const data = await response.json();

      if (response.ok) {
        onVerify(data.token);
      } else {
        setError(data.error || 'Invalid code');
      }
    } catch (err) {
      setError('Network error. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="two-factor-auth">
      <div className="auth-card">
        <h2>Two-Factor Authentication</h2>
        <p>Enter the 6-digit code from your authenticator app</p>
        
        <form onSubmit={handleSubmit}>
          <div className="form-group">
            <input
              type="text"
              value={code}
              onChange={(e) => setCode(e.target.value.replace(/\D/g, '').slice(0, 6))}
              placeholder="000000"
              maxLength="6"
              required
            />
          </div>

          {error && <div className="error-message">{error}</div>}

          <div className="auth-actions">
            <button type="button" onClick={onCancel} className="cancel-button">
              Cancel
            </button>
            <button type="submit" disabled={loading || code.length !== 6} className="verify-button">
              {loading ? 'Verifying...' : 'Verify'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

export default TwoFactorAuth;
