import React, { SyntheticEvent } from 'react';
import { useAppContext } from '../../providers/useAppContext';
import { Snackbar } from '@mui/joy';
import Button from '@mui/joy/Button';
import Info from '@mui/icons-material/Info';

const InfoBanner: React.FC = () => {
    const { info } = useAppContext();
    const [open, setOpen] = React.useState(false);

    React.useEffect(() => {
        if (info) {
            setOpen(true);
        }
    }, [info]);

    const handleClose = (event: SyntheticEvent | Event, reason?: string) => {
        if (reason !== 'clickaway') {
            setOpen(false);
        }
    };

    return info?.message ? (
        <React.Fragment>
            <Snackbar
                variant="solid"
                color="primary"
                open={open}
                anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
                startDecorator={<Info />}
                endDecorator={
                    <Button onClick={handleClose} size="sm" variant="solid" color="primary">
                        Ok
                    </Button>
                }
            >
                {(info?.message.toString()) || 'Unknown error'}
            </Snackbar>
        </React.Fragment>
    ) : null
};

export default InfoBanner;
