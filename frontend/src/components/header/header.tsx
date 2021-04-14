import { FC } from 'react';

import PropTypes from 'prop-types';

import '../../assets/css/header.css';

interface HeaderProps {
  title: string
}

const Header: FC<HeaderProps> = ({ title }) => (
  <header id="header">
    <div className="title">
      { title }
    </div>
  </header>
);
Header.propTypes = {
  title: PropTypes.string.isRequired,
};

export default Header;
