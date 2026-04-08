import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

const sidebars: SidebarsConfig = {
  manualSidebar: [
    'intro',
    {
      type: 'category',
      label: 'Part 1: Getting Started',
      items: [
        'getting-started/introduction',
        'getting-started/installation',
        'getting-started/dashboard',
        {
          type: 'category',
          label: 'Installation Guides',
          items: [
            'getting-started/install-guide-compose',
            'getting-started/install-guide-k8s',
          ],
        },
      ],
    },
    {
      type: 'category',
      label: 'Part 2: Core Workflow',
      items: [
        'core-workflow/researcher-journey',
        'core-workflow/data-model-management',
        'core-workflow/sae-training',
        'core-workflow/feature-extraction',
        'core-workflow/auto-labeling',
        'core-workflow/steering',
      ],
    },
    {
      type: 'category',
      label: 'Part 3: Advanced Usage',
      items: [
        'advanced/templates',
        'advanced/external-saes',
        'advanced/multi-gpu',
        'advanced/exporting',
        'advanced/multi-dataset',
      ],
    },
    'troubleshooting',
  ],
};

export default sidebars;
