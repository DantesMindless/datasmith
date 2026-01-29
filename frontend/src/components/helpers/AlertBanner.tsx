import React, { SyntheticEvent } from 'react';
import { useAppContext } from '../../providers/useAppContext';
import { Snackbar } from '@mui/joy';
import Button from '@mui/joy/Button';
import Dangerous from '@mui/icons-material/Dangerous';
import CheckCircle from '@mui/icons-material/CheckCircle';
import Warning from '@mui/icons-material/Warning';
import Info from '@mui/icons-material/Info';

const AlertBanner: React.FC = () => {
    const { alert } = useAppContext();
    const [open, setOpen] = React.useState(false);

    React.useEffect(() => {
        if (alert) {
            setOpen(true);
        }
    }, [alert]);

    const handleClose = (event: SyntheticEvent | Event, reason?: string) => {
        if (reason !== 'clickaway') {
            setOpen(false);
        }
    };

    // Determine color and icon based on alert type
    const getAlertProps = () => {
        switch (alert?.type) {
            case 'success':
                return {
                    color: 'primary' as const,
                    icon: <CheckCircle />,
                    buttonColor: 'primary' as const
                };
            case 'warning':
                return {
                    color: 'warning' as const,
                    icon: <Warning />,
                    buttonColor: 'warning' as const
                };
            case 'info':
                return {
                    color: 'neutral' as const,
                    icon: <Info />,
                    buttonColor: 'neutral' as const
                };
            case 'error':
            default:
                return {
                    color: 'danger' as const,
                    icon: <Dangerous />,
                    buttonColor: 'danger' as const
                };
        }
    };

    const alertProps = getAlertProps();

    return alert?.message ? (
        <React.Fragment>
            <Snackbar
                variant="solid"
                color={alertProps.color}
                open={open}
                anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
                startDecorator={alertProps.icon}
                endDecorator={
                    <Button onClick={handleClose} size="sm" variant="soft" color={alertProps.buttonColor}>
                        Dismiss
                    </Button>
                }
            >
                {(alert?.message.toString()) || 'Unknown error'}
            </Snackbar>
        </React.Fragment>
    ) : null
};

export default AlertBanner;
