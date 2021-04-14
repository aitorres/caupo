import React, { FC } from 'react';
import PropTypes from 'prop-types';

import '../../assets/css/box.css';

interface BoxProps {
  children: React.ReactChild,
  id?: string
}

const Box: FC<BoxProps> = ({ id, children }) => (
  <div className="box" id={id}>
    { children }
  </div>
);
Box.propTypes = {
  children: PropTypes.element.isRequired,
  id: PropTypes.string,
};
Box.defaultProps = {
  id: '',
};

export default Box;
