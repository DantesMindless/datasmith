import React, { SyntheticEvent } from 'react';
import { useAlert } from '../../providers/UseAlert';
import { Snackbar, Alert } from '@mui/joy';
import Button from '@mui/joy/Button';
import PlaylistAddCheckCircleRoundedIcon from '@mui/icons-material/PlaylistAddCheckCircleRounded';

const AlertBanner: React.FC = () => {
    const { alert } = useAlert();
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

    return alert?.message ? (
        <React.Fragment>
            <Snackbar
                variant="solid"
                color="danger"
                open={open}
                anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
                startDecorator={<PlaylistAddCheckCircleRoundedIcon />}
                endDecorator={
                    <Button onClick={() => setOpen(false)} size="sm" variant="soft" color="danger">
                        Dismiss
                    </Button>
                }
            >
                {alert?.message || 'Unknown error'}
            </Snackbar>
        </React.Fragment>
    ) : null
};

export default AlertBanner;
