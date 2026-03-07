import {
  Database,
  Server,
  GraduationCap,
  Layers,
  Tags,
  Network,
  Sliders,
  FileText,
  Activity,
  ChevronLeft,
  ChevronRight,
} from 'lucide-react';
import logoSvg from '../../assets/logo.svg';
import { useUIStore } from '../../stores/uiStore';

type ActivePanel = 'datasets' | 'models' | 'training' | 'extractions' | 'labeling' | 'saes' | 'steering' | 'templates' | 'system';

const navItems: { id: ActivePanel; label: string; icon: typeof Database }[] = [
  { id: 'datasets', label: 'Datasets', icon: Database },
  { id: 'models', label: 'Models', icon: Server },
  { id: 'training', label: 'Training', icon: GraduationCap },
  { id: 'extractions', label: 'Extractions', icon: Layers },
  { id: 'labeling', label: 'Labeling', icon: Tags },
  { id: 'saes', label: 'SAEs', icon: Network },
  { id: 'steering', label: 'Steering', icon: Sliders },
  { id: 'templates', label: 'Templates', icon: FileText },
  { id: 'system', label: 'Monitor', icon: Activity },
];

interface SidebarProps {
  activePanel: ActivePanel;
  onPanelChange: (panel: ActivePanel) => void;
}

export function Sidebar({ activePanel, onPanelChange }: SidebarProps) {
  const { sidebar, toggleSidebar } = useUIStore();
  const { collapsed } = sidebar;

  return (
    <aside
      className={`
        fixed left-0 top-0 h-screen
        bg-white dark:bg-slate-900/95 border-r border-slate-200 dark:border-slate-700/50
        backdrop-blur-sm z-40
        transition-all duration-300 ease-in-out
        ${collapsed ? 'w-16' : 'w-56'}
      `}
    >
      {/* Logo */}
      <div className="h-14 flex items-center px-4 border-b border-slate-200 dark:border-slate-700/50">
        <div className="flex items-center gap-3">
          <img
            src={logoSvg}
            alt="MechInterp Studio"
            className="w-8 h-8 flex-shrink-0"
          />
          {!collapsed && (
            <div className="overflow-hidden">
              <div className="text-base font-semibold text-slate-900 dark:text-slate-100">MechInterp Studio</div>
              <div className="text-[10px] text-slate-500 dark:text-slate-400 leading-tight">
                Edge AI Feature Discovery
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Navigation */}
      <nav className="p-2 space-y-0.5">
        {navItems.map((item) => (
          <button
            key={item.id}
            onClick={() => onPanelChange(item.id)}
            className={`
              w-full flex items-center gap-3 px-3 py-2.5 rounded-lg
              transition-all duration-200
              ${activePanel === item.id
                ? 'bg-emerald-500/10 text-emerald-600 dark:text-emerald-400'
                : 'text-slate-600 dark:text-slate-400 hover:bg-slate-100 dark:hover:bg-slate-800 hover:text-slate-900 dark:hover:text-slate-200'
              }
              ${collapsed ? 'justify-center' : ''}
            `}
            title={collapsed ? item.label : undefined}
          >
            <item.icon className="w-5 h-5 flex-shrink-0" />
            {!collapsed && (
              <span className="text-sm font-medium">{item.label}</span>
            )}
          </button>
        ))}
      </nav>

      {/* Collapse Toggle */}
      <button
        onClick={toggleSidebar}
        className="absolute -right-3 top-20 w-6 h-6 bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded-full flex items-center justify-center text-slate-400 hover:text-slate-600 dark:hover:text-slate-200 hover:bg-slate-50 dark:hover:bg-slate-700 transition-colors"
      >
        {collapsed ? (
          <ChevronRight className="w-3 h-3" />
        ) : (
          <ChevronLeft className="w-3 h-3" />
        )}
      </button>
    </aside>
  );
}
