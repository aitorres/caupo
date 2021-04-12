module.exports = {
  parser: "@typescript-eslint/parser",
  plugins: ["@typescript-eslint"],
  ignorePatterns: [
    "reportWebVitals.ts",
  ],
  rules: {
    "react/react-in-jsx-scope": "off",
    "max-len": [2, {
      "code": 120,
    }],
    "jsx-a11y/label-has-associated-control": [ 2, {
      "assert": "either",
    }]
  },
  extends: [
    "eslint:recommended",
    "airbnb-typescript",
    "plugin:@typescript-eslint/recommended",
    "plugin:@typescript-eslint/recommended-requiring-type-checking"
  ],
  parserOptions: {
    project: './tsconfig.json'
  }
}
