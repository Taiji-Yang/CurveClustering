import { Modal } from 'react-bootstrap';
import { Buttonb } from 'react-bootstrap';
import Image from 'react-bootstrap/Image'
import {  useState, useEffect } from 'react'
import 'bootstrap/dist/css/bootstrap.min.css';
import esc from './esc.png'
import Button from '@material-ui/core/Button';
import React from 'react';
import PropTypes from 'prop-types';
import { makeStyles } from '@material-ui/core/styles';
import Tabs from '@material-ui/core/Tabs';
import Tab from '@material-ui/core/Tab';
import Typography from '@material-ui/core/Typography';
import Box from '@material-ui/core/Box';
import Switch from '@material-ui/core/Switch';
import FormGroup from '@material-ui/core/FormGroup';
import FormControlLabel from '@material-ui/core/FormControlLabel';

function TabPanel(props) {
    const { children, value, index, ...other } = props;
  
    return (
      <div
        role="tabpanel"
        hidden={value !== index}
        id={`vertical-tabpanel-${index}`}
        aria-labelledby={`vertical-tab-${index}`}
        {...other}
      >
        {value === index && (
          <Box p={3}>
            <Typography>{children}</Typography>
          </Box>
        )}
      </div>
    );
  }
  
  TabPanel.propTypes = {
    children: PropTypes.node,
    index: PropTypes.any.isRequired,
    value: PropTypes.any.isRequired,
  };
  
  function a11yProps(index) {
    return {
      id: `vertical-tab-${index}`,
      'aria-controls': `vertical-tabpanel-${index}`,
    };
  }
  
  const useStyles = makeStyles((theme) => ({
    root: {
      flexGrow: 1,
      backgroundColor: theme.palette.background.paper,
      display: 'flex',
      height: 224,
    },
    tabs: {
      borderRight: `1px solid ${theme.palette.divider}`,
    },
  }));

const MyVerticallyCenteredModal = (props) => {

    function getNone(){
        fetch('/deleteall')
        window.location.reload(false)
    }
    const classes = useStyles();
    const [value, setValue] = useState(0);
    const [state, setSwitchState] = useState({
        localcenter: false,

    });

    const handleChange = (event, newValue) => {
        setValue(newValue);
    };
    const handleSwitchChange = (event) => {
        setSwitchState({...state, [event.target.name]: event.target.checked})
    };

    return (
        
      <Modal
        {...props}
        size="lg"
        aria-labelledby="contained-modal-title-vcenter"
        centered
      >
        <Modal.Header>
          <Modal.Title id="contained-modal-title-vcenter">
            Settings
          </Modal.Title>
        </Modal.Header>
        <Modal.Body>
        <div className={classes.root}>
            <Tabs
                orientation="vertical"
                variant="scrollable"
                value={value}
                onChange={handleChange}
                aria-label="Vertical tabs example"
                className={classes.tabs}
            >
                <Tab label="data" {...a11yProps(0)} />
                <Tab label="clustering" {...a11yProps(1)} />
                <Tab label="algorithm" {...a11yProps(2)} />
                <Tab label="threshold" {...a11yProps(3)} />
                <Tab label="options" {...a11yProps(4)} />
            </Tabs>
            <TabPanel value={value} index={0}>
                <FormGroup row>
                    <FormControlLabel
                        control={
                            <Switch
                                checked={state.localcenter}
                                onChange={handleSwitchChange}
                                name="localcenter"
                                color='secondary'
                            />
                        }
                        label="test"
                    />
                </FormGroup>
            </TabPanel>
            <TabPanel value={value} index={1}>
                Item Two
            </TabPanel>
            <TabPanel value={value} index={2}>
                Item Three
            </TabPanel>
            <TabPanel value={value} index={3}>
                Item Four
            </TabPanel>
            <TabPanel value={value} index={4}>
                Item Five
            </TabPanel>
         </div>
         <div style = {{textAlign: 'center'}}>
            <Button 
                variant="outlined" 
                color="secondary" 
                style = {{}}
                onClick={getNone}
            >
                clear canvas
            </Button>
         </div>
        </Modal.Body>
        <Modal.Footer>
          <Modal.Title id="contained-modal-title-vcenter" size = 'sm'>
            Press <Image src={esc} rounded style = {{width: '40px', height: '40px'}}></Image> to close
          </Modal.Title>
        </Modal.Footer>
      </Modal>
    );
  }

  export default MyVerticallyCenteredModal
