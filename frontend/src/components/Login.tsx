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
  DataObject,
} from '@mui/icons-material';
import { useAppContext } from '../providers/useAppContext';

interface LoginProps {
  onSwitchToSignup?: () => void;
}

const Login: React.FC<LoginProps> = ({ onSwitchToSignup }) => {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState('');
  const { login } = useAppContext();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!username || !password) {
      setError('Please fill in all fields');
      return;
    }
    
    setLoading(true);
    setError('');
    try {
      await login(username, password);
    } catch (err) {
      setError('Invalid credentials. Please try again.');
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
                  Welcome to DataSmith
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Sign in to access your data analytics platform
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
              
              {/* Login Form */}
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
                  sx={{ mb: 2 }}
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
                  sx={{ mb: 3 }}
                />
                
                <Button
                  type="submit"
                  fullWidth
                  variant="contained"
                  size="large"
                  disabled={loading || !username || !password}
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
                  {loading ? 'Signing In...' : 'Sign In'}
                </Button>
              </Box>

              {/* Footer with Signup Link */}
              <Box sx={{ textAlign: 'center', mt: 3 }}>
                <Typography variant="body2" color="text.secondary">
                  Don't have an account?{' '}
                  <Link
                    component="button"
                    variant="body2"
                    onClick={onSwitchToSignup}
                    sx={{
                      fontWeight: 600,
                      cursor: 'pointer',
                      textDecoration: 'none',
                      '&:hover': {
                        textDecoration: 'underline',
                      }
                    }}
                  >
                    Sign Up
                  </Link>
                </Typography>
              </Box>

              <Box sx={{ textAlign: 'center', mt: 2 }}>
                <Typography variant="caption" color="text.secondary">
                  Secure access to your data analytics workspace
                </Typography>
              </Box>
            </CardContent>
          </Card>
        </Fade>
      </Container>
    </Box>
  );
};

export default Login;