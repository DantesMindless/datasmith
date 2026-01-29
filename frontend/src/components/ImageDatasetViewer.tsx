import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Grid,
  Card,
  CardMedia,
  CardContent,
  CircularProgress,
  Alert,
  Pagination,
  Stack,
  Chip,
  IconButton,
  Dialog,
  DialogContent,
  DialogTitle,
  DialogActions,
  Button,
  TextField,
  InputAdornment,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
} from '@mui/material';
import {
  ZoomIn,
  Close,
  Search,
  Image as ImageIcon,
  FilterList,
} from '@mui/icons-material';
import httpfetch, { getAccessToken } from '../utils/axios';

interface ImageFile {
  filename: string;
  path: string;
  size: number;
  extension: string;
  class?: string;  // Classification label/category
}

interface ImageDatasetViewerProps {
  datasetId: string | number;
}

// Global cache for blob URLs to avoid refetching
const imageBlobCache = new Map<string, string>();

// Authenticated image component that fetches with JWT token
const AuthenticatedImage: React.FC<{
  datasetId: string | number;
  imagePath: string;
  alt: string;
  height?: number | string;
  style?: React.CSSProperties;
  sx?: any;
}> = ({ datasetId, imagePath, alt, height, style, sx }) => {
  const [imageUrl, setImageUrl] = useState<string>('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);
  const imgRef = React.useRef<HTMLDivElement>(null);

  useEffect(() => {
    const cacheKey = `${datasetId}:${imagePath}`;

    // Check if already cached
    if (imageBlobCache.has(cacheKey)) {
      setImageUrl(imageBlobCache.get(cacheKey)!);
      setLoading(false);
      return;
    }

    let isSubscribed = true;

    // Use IntersectionObserver for lazy loading
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting && isSubscribed) {
            fetchImage();
            observer.disconnect();
          }
        });
      },
      { rootMargin: '100px' } // Start loading 100px before image is visible
    );

    if (imgRef.current) {
      observer.observe(imgRef.current);
    }

    const fetchImage = async () => {
      try {
        if (!isSubscribed) return;

        setLoading(true);
        setError(false);

        const baseURL = import.meta.env.VITE_API_URL || "http://localhost:8000/api/";
        const token = getAccessToken();

        const response = await fetch(`${baseURL}datasets/${datasetId}/images/${imagePath}`, {
          headers: {
            'Authorization': `Bearer ${token}`,
          },
        });

        if (!response.ok) {
          throw new Error('Failed to load image');
        }

        const blob = await response.blob();
        const url = URL.createObjectURL(blob);

        if (isSubscribed) {
          // Cache the blob URL
          imageBlobCache.set(cacheKey, url);
          setImageUrl(url);
        }
      } catch (err) {
        console.error('Error loading image:', err);
        if (isSubscribed) {
          setError(true);
        }
      } finally {
        if (isSubscribed) {
          setLoading(false);
        }
      }
    };

    return () => {
      isSubscribed = false;
      observer.disconnect();
      // Don't revoke blob URLs - keep them cached for better performance
      // They'll be cleaned up when the page is closed
    };
  }, [datasetId, imagePath]);

  if (loading) {
    return (
      <Box
        ref={imgRef}
        sx={{
          height: height || 200,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          bgcolor: 'grey.100',
          ...sx,
        }}
      >
        <CircularProgress size={24} />
      </Box>
    );
  }

  if (error || !imageUrl) {
    return (
      <Box
        ref={imgRef}
        sx={{
          height: height || 200,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          bgcolor: 'grey.100',
          ...sx,
        }}
      >
        <ImageIcon sx={{ fontSize: 40, color: 'text.disabled' }} />
      </Box>
    );
  }

  return (
    <Box
      component="img"
      src={imageUrl}
      alt={alt}
      sx={{
        height: height || 200,
        width: '100%',
        objectFit: 'cover',
        bgcolor: 'grey.100',
        ...sx,
      }}
      style={style}
    />
  );
};

const ImageDatasetViewer: React.FC<ImageDatasetViewerProps> = ({ datasetId }) => {
  const [images, setImages] = useState<ImageFile[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(8);
  const [totalPages, setTotalPages] = useState(1);
  const [totalImages, setTotalImages] = useState(0);
  const [selectedImage, setSelectedImage] = useState<ImageFile | null>(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [filterExtension, setFilterExtension] = useState('');
  const [filterClass, setFilterClass] = useState('');

  const fetchImages = async (currentPage: number = 1) => {
    try {
      setLoading(true);
      setError('');

      const response = await httpfetch.get(`datasets/${datasetId}/images/?page=${currentPage}&page_size=${pageSize}`);

      setImages(response.data.images || []);
      setTotalPages(response.data.total_pages || 1);
      setTotalImages(response.data.total_images || 0);
    } catch (err: any) {
      console.error('Error fetching images:', err);
      setError(err.response?.data?.error || 'Failed to load images');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchImages(page);
  }, [datasetId, page, pageSize]);

  const handlePageChange = (event: React.ChangeEvent<unknown>, value: number) => {
    setPage(value);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  const handleImageClick = (image: ImageFile) => {
    setSelectedImage(image);
  };

  const handleCloseDialog = () => {
    setSelectedImage(null);
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  // Get unique extensions and classes for filters
  const uniqueExtensions = [...new Set(images.map(img => img.extension))];
  const uniqueClasses = [...new Set(images.map(img => img.class).filter(Boolean))];

  // Filter images based on search, extension, and class
  // Note: Client-side filtering is applied to already paginated results from server
  const filteredImages = images.filter(image => {
    const matchesSearch = searchTerm === '' ||
      image.filename.toLowerCase().includes(searchTerm.toLowerCase()) ||
      image.path.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesExtension = filterExtension === '' ||
      image.extension === filterExtension;
    const matchesClass = filterClass === '' ||
      image.class === filterClass;
    return matchesSearch && matchesExtension && matchesClass;
  });

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: 400 }}>
        <Stack alignItems="center" spacing={2}>
          <CircularProgress />
          <Typography variant="body2" color="text.secondary">
            Loading images...
          </Typography>
        </Stack>
      </Box>
    );
  }

  if (error) {
    return (
      <Alert severity="error" sx={{ mb: 3 }}>
        {error}
      </Alert>
    );
  }

  if (images.length === 0) {
    return (
      <Box sx={{ textAlign: 'center', py: 8 }}>
        <ImageIcon sx={{ fontSize: 64, color: 'text.disabled', mb: 2 }} />
        <Typography variant="h6" color="text.secondary">
          No images found in this dataset
        </Typography>
      </Box>
    );
  }

  return (
    <Box>
      {/* Header with stats and filters */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Grid container spacing={3} alignItems="center">
            <Grid item xs={12} md={4}>
              <Stack direction="row" spacing={2} alignItems="center">
                <ImageIcon sx={{ fontSize: 40, color: 'primary.main' }} />
                <Box>
                  <Typography variant="h6" fontWeight={600}>
                    {totalImages} Images
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    Page {page} of {totalPages}
                  </Typography>
                </Box>
              </Stack>
            </Grid>
            <Grid item xs={12} md={4}>
              <TextField
                fullWidth
                size="small"
                placeholder="Search images..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                InputProps={{
                  startAdornment: (
                    <InputAdornment position="start">
                      <Search />
                    </InputAdornment>
                  ),
                }}
              />
            </Grid>
            <Grid item xs={12} md={2}>
              <FormControl fullWidth size="small">
                <InputLabel>Class</InputLabel>
                <Select
                  value={filterClass}
                  label="Class"
                  onChange={(e) => setFilterClass(e.target.value)}
                >
                  <MenuItem value="">All Classes</MenuItem>
                  {uniqueClasses.map(cls => (
                    <MenuItem key={cls} value={cls}>
                      {cls}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} md={2}>
              <FormControl fullWidth size="small">
                <InputLabel>Extension</InputLabel>
                <Select
                  value={filterExtension}
                  label="Extension"
                  onChange={(e) => setFilterExtension(e.target.value)}
                >
                  <MenuItem value="">All</MenuItem>
                  {uniqueExtensions.map(ext => (
                    <MenuItem key={ext} value={ext}>
                      {ext.toUpperCase()}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} md={2}>
              <FormControl fullWidth size="small">
                <InputLabel>Per Page</InputLabel>
                <Select
                  value={pageSize}
                  label="Per Page"
                  onChange={(e) => {
                    setPageSize(Number(e.target.value));
                    setPage(1);
                  }}
                >
                  <MenuItem value={8}>8</MenuItem>
                  <MenuItem value={16}>16</MenuItem>
                  <MenuItem value={24}>24</MenuItem>
                  <MenuItem value={48}>48</MenuItem>
                </Select>
              </FormControl>
            </Grid>
          </Grid>
        </CardContent>
      </Card>

      {/* Image Grid */}
      <Grid container spacing={2}>
        {filteredImages.map((image, index) => (
          <Grid item xs={12} sm={6} md={4} lg={3} key={index}>
            <Card
              sx={{
                cursor: 'pointer',
                transition: 'all 0.3s ease',
                '&:hover': {
                  transform: 'translateY(-4px)',
                  boxShadow: 4,
                },
              }}
              onClick={() => handleImageClick(image)}
            >
              <AuthenticatedImage
                datasetId={datasetId}
                imagePath={image.path}
                alt={image.filename}
                height={200}
              />
              <CardContent sx={{ p: 1.5 }}>
                {image.class && (
                  <Chip
                    label={image.class}
                    size="small"
                    color="primary"
                    sx={{ fontSize: '0.7rem', height: 20, mb: 0.5, fontWeight: 600 }}
                  />
                )}
                <Typography
                  variant="caption"
                  fontWeight={500}
                  sx={{
                    display: 'block',
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                    whiteSpace: 'nowrap',
                  }}
                  title={image.path}
                >
                  {image.path}
                </Typography>
                <Stack direction="row" spacing={1} sx={{ mt: 0.5 }}>
                  <Chip
                    label={image.extension.toUpperCase()}
                    size="small"
                    sx={{ fontSize: '0.65rem', height: 18 }}
                  />
                  <Chip
                    label={formatFileSize(image.size)}
                    size="small"
                    sx={{ fontSize: '0.65rem', height: 18 }}
                    variant="outlined"
                  />
                </Stack>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>

      {/* Pagination */}
      {totalPages > 1 && (
        <Box sx={{ display: 'flex', justifyContent: 'center', mt: 4 }}>
          <Pagination
            count={totalPages}
            page={page}
            onChange={handlePageChange}
            color="primary"
            size="large"
            showFirstButton
            showLastButton
          />
        </Box>
      )}

      {/* Image Preview Dialog */}
      <Dialog
        open={selectedImage !== null}
        onClose={handleCloseDialog}
        maxWidth="lg"
        fullWidth
      >
        {selectedImage && (
          <>
            <DialogTitle sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <Box>
                <Typography variant="h6" fontWeight={600}>
                  {selectedImage.path}
                </Typography>
                <Stack direction="row" spacing={1} sx={{ mt: 1 }}>
                  {selectedImage.class && (
                    <Chip label={selectedImage.class} size="small" color="primary" />
                  )}
                  <Chip label={selectedImage.extension.toUpperCase()} size="small" />
                  <Chip label={formatFileSize(selectedImage.size)} size="small" variant="outlined" />
                </Stack>
              </Box>
              <IconButton onClick={handleCloseDialog}>
                <Close />
              </IconButton>
            </DialogTitle>
            <DialogContent>
              <AuthenticatedImage
                datasetId={datasetId}
                imagePath={selectedImage.path}
                alt={selectedImage.filename}
                style={{
                  maxWidth: '100%',
                  maxHeight: '70vh',
                  objectFit: 'contain',
                }}
                sx={{
                  width: 'auto',
                  height: 'auto',
                  maxHeight: '70vh',
                  objectFit: 'contain',
                  borderRadius: 1,
                }}
              />
            </DialogContent>
            <DialogActions>
              <Button onClick={handleCloseDialog}>Close</Button>
            </DialogActions>
          </>
        )}
      </Dialog>
    </Box>
  );
};

export default ImageDatasetViewer;
