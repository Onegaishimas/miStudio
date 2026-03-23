import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';

const config: Config = {
  title: 'miStudio Manual',
  tagline: 'MechInterp Studio — End-to-End Mechanistic Interpretability Platform',
  favicon: 'img/favicon.ico',

  // GitHub Pages deployment
  url: 'https://onegaishimas.github.io',
  baseUrl: '/miStudio/',

  // GitHub Pages config
  organizationName: 'Onegaishimas',
  projectName: 'miStudio',
  deploymentBranch: 'gh-pages',
  trailingSlash: false,

  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',

  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      {
        docs: {
          routeBasePath: '/',
          sidebarPath: './sidebars.ts',
          editUrl: 'https://github.com/Onegaishimas/miStudio/tree/main/manual/',
        },
        blog: false,
        theme: {
          customCss: './src/css/custom.css',
        },
      } satisfies Preset.Options,
    ],
  ],

  themeConfig: {
    navbar: {
      title: 'miStudio Manual',
      items: [
        {
          type: 'docSidebar',
          sidebarId: 'manualSidebar',
          position: 'left',
          label: 'Manual',
        },
        {
          href: 'https://github.com/Onegaishimas/miStudio',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Manual',
          items: [
            {label: 'Getting Started', to: '/getting-started/introduction'},
            {label: 'Core Workflow', to: '/core-workflow/researcher-journey'},
            {label: 'Troubleshooting', to: '/troubleshooting'},
          ],
        },
        {
          title: 'Resources',
          items: [
            {label: 'GitHub', href: 'https://github.com/Onegaishimas/miStudio'},
            {label: 'Neuronpedia', href: 'https://neuronpedia.org'},
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} MCS Lab. Built with Docusaurus.`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
      additionalLanguages: ['bash', 'python', 'typescript', 'nginx', 'yaml', 'docker'],
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
