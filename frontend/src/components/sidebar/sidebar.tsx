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
      <li>
        <a href="/#">¿Qué es esto?</a>
      </li>
      <li>
        <Link to="/entities">Entidades</Link>
      </li>
      <li>
        <Link to="/entity-variation">Variación de Entidades</Link>
      </li>
    </ul>
  </Box>
);

export default Sidebar;
