import React, { useState } from 'react';
import {
  Box,
  Button,
  TextField,
  Typography,
  Card,
  CardContent,
  Container,
  Alert,
  CircularProgress,
  InputAdornment,
  IconButton,
  Fade,
  Link,
} from '@mui/material';
import {
  Visibility,
  VisibilityOff,
  Person,
  Email,
  DataObject,
} from '@mui/icons-material';

interface SignupProps {
  onSwitchToLogin: () => void;
  onSignupSuccess: () => void;
}

const Signup: React.FC<SignupProps> = ({ onSwitchToLogin, onSignupSuccess }) => {
  const [username, setUsername] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const [hasSubmitted, setHasSubmitted] = useState(false);

  const validateForm = (): boolean => {
    if (!username || !email || !password || !confirmPassword) {
      setError('Please fill in all fields');
      return false;
    }
    if (password !== confirmPassword) {
      setError('Passwords do not match');
      return false;
    }
    if (password.length < 8) {
      setError('Password must be at least 8 characters');
      return false;
    }
    if (!email.includes('@')) {
      setError('Please enter a valid email address');
      return false;
    }
    return true;
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setSuccess('');
    setHasSubmitted(true);

    if (!validateForm()) {
      return;
    }

    setLoading(true);
    try {
      const baseURL = import.meta.env.VITE_API_URL || "http://localhost:8000/api/";
      const response = await fetch(`${baseURL}signup/`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          username,
          email,
          password,
        }),
      });

      const data = await response.json();

      if (response.ok) {
        setSuccess('Account created successfully! Redirecting to login...');
        setTimeout(() => {
          onSignupSuccess();
        }, 1500);
      } else {
        // Handle specific error messages from backend
        if (data.username) {
          setError(`Username: ${data.username.join(', ')}`);
        } else if (data.email) {
          setError(`Email: ${data.email.join(', ')}`);
        } else if (data.password) {
          setError(`Password: ${data.password.join(', ')}`);
        } else if (data.error) {
          setError(data.error);
        } else {
          setError('Registration failed. Please try again.');
        }
      }
    } catch (err) {
      setError('Network error. Please check your connection.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box
      sx={{
        minHeight: '100vh',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        bgcolor: 'background.default',
        p: 2,
        background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
        position: 'relative',
        '&::before': {
          content: '""',
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          bgcolor: 'background.default',
          opacity: 0.9,
          zIndex: 0,
        }
      }}
    >
      <Container maxWidth="sm" sx={{ position: 'relative', zIndex: 1 }}>
        <Fade in timeout={800}>
          <Card
            sx={{
              maxWidth: 420,
              mx: 'auto',
              borderRadius: 3,
              boxShadow: '0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04)',
              backdropFilter: 'blur(10px)',
              border: '1px solid',
              borderColor: 'divider',
            }}
          >
            <CardContent sx={{ p: 5 }}>
              {/* Logo and Brand */}
              <Box sx={{ textAlign: 'center', mb: 4 }}>
                <Box
                  sx={{
                    width: 64,
                    height: 64,
                    borderRadius: 3,
                    bgcolor: 'primary.main',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    mx: 'auto',
                    mb: 2,
                    boxShadow: '0 10px 15px -3px rgba(37, 99, 235, 0.3)',
                  }}
                >
                  <DataObject sx={{ color: 'white', fontSize: 32 }} />
                </Box>
                <Typography variant="h4" fontWeight={700} gutterBottom>
                  Create Account
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Sign up to start using DataSmith
                </Typography>
              </Box>

              {/* Error Alert */}
              {error && (
                <Fade in>
                  <Alert
                    severity="error"
                    sx={{ mb: 3, borderRadius: 2 }}
                    onClose={() => setError('')}
                  >
                    {error}
                  </Alert>
                </Fade>
              )}

              {/* Success Alert */}
              {success && (
                <Fade in>
                  <Alert
                    severity="success"
                    sx={{ mb: 3, borderRadius: 2 }}
                  >
                    {success}
                  </Alert>
                </Fade>
              )}

              {/* Signup Form */}
              <Box component="form" onSubmit={handleSubmit}>
                <TextField
                  fullWidth
                  label="Username"
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  margin="normal"
                  required
                  autoFocus
                  disabled={loading}
                  InputProps={{
                    startAdornment: (
                      <InputAdornment position="start">
                        <Person color="action" />
                      </InputAdornment>
                    ),
                  }}
                  sx={{ mb: 1 }}
                />

                <TextField
                  fullWidth
                  label="Email"
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  margin="normal"
                  required
                  disabled={loading}
                  InputProps={{
                    startAdornment: (
                      <InputAdornment position="start">
                        <Email color="action" />
                      </InputAdornment>
                    ),
                  }}
                  sx={{ mb: 1 }}
                />

                <TextField
                  fullWidth
                  label="Password"
                  type={showPassword ? 'text' : 'password'}
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  margin="normal"
                  required
                  disabled={loading}
                  helperText="At least 8 characters"
                  InputProps={{
                    endAdornment: (
                      <InputAdornment position="end">
                        <IconButton
                          onClick={() => setShowPassword(!showPassword)}
                          edge="end"
                          disabled={loading}
                        >
                          {showPassword ? <VisibilityOff /> : <Visibility />}
                        </IconButton>
                      </InputAdornment>
                    ),
                  }}
                  sx={{ mb: 1 }}
                />

                <TextField
                  fullWidth
                  label="Confirm Password"
                  type={showConfirmPassword ? 'text' : 'password'}
                  value={confirmPassword}
                  onChange={(e) => setConfirmPassword(e.target.value)}
                  margin="normal"
                  required
                  disabled={loading}
                  error={hasSubmitted && confirmPassword !== '' && password !== confirmPassword}
                  helperText={hasSubmitted && confirmPassword !== '' && password !== confirmPassword ? 'Passwords do not match' : ''}
                  InputProps={{
                    endAdornment: (
                      <InputAdornment position="end">
                        <IconButton
                          onClick={() => setShowConfirmPassword(!showConfirmPassword)}
                          edge="end"
                          disabled={loading}
                        >
                          {showConfirmPassword ? <VisibilityOff /> : <Visibility />}
                        </IconButton>
                      </InputAdornment>
                    ),
                  }}
                  sx={{ mb: 3 }}
                />

                <Button
                  type="submit"
                  fullWidth
                  variant="contained"
                  size="large"
                  disabled={loading || !username || !email || !password || !confirmPassword}
                  sx={{
                    py: 1.5,
                    borderRadius: 2,
                    fontWeight: 600,
                    fontSize: '1rem',
                    textTransform: 'none',
                    boxShadow: '0 4px 12px rgba(37, 99, 235, 0.3)',
                    '&:hover': {
                      boxShadow: '0 8px 20px rgba(37, 99, 235, 0.4)',
                    },
                  }}
                  startIcon={loading ? <CircularProgress size={20} color="inherit" /> : undefined}
                >
                  {loading ? 'Creating Account...' : 'Create Account'}
                </Button>
              </Box>

              {/* Footer with Login Link */}
              <Box sx={{ textAlign: 'center', mt: 3 }}>
                <Typography variant="body2" color="text.secondary">
                  Already have an account?{' '}
                  <Link
                    component="button"
                    variant="body2"
                    onClick={onSwitchToLogin}
                    sx={{
                      fontWeight: 600,
                      cursor: 'pointer',
                      textDecoration: 'none',
                      '&:hover': {
                        textDecoration: 'underline',
                      }
                    }}
                  >
                    Sign In
                  </Link>
                </Typography>
              </Box>
            </CardContent>
          </Card>
        </Fade>
      </Container>
    </Box>
  );
};

export default Signup;
