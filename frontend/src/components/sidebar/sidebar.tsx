import { FC } from 'react';
import { Link } from 'react-router-dom';

import Box from '../box/box';

import '../../assets/css/sidebar.css';

const Sidebar: FC = () => (
  <Box id="sidebar">
    <ul>
      <li>
        <Link to="/">Inicio</Link>
      </li>
    </ul>
  </Box>
);

export default Sidebar;
