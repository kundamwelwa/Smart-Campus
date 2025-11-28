"""
Plugin System for Hot-Loading Extensible Modules

Implements plugin architecture with dependency injection, lifecycle management,
and hot-reload capabilities without system restart.
"""

import importlib
import importlib.util
import inspect
import sys
from abc import ABC, abstractmethod
from collections.abc import Callable
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)


class PluginStatus(str, Enum):
    """Plugin lifecycle status."""

    UNLOADED = "unloaded"
    LOADING = "loading"
    LOADED = "loaded"
    ACTIVE = "active"
    ERROR = "error"
    UNLOADING = "unloading"


class PluginMetadata(BaseModel):
    """Plugin metadata and configuration."""

    id: UUID = Field(default_factory=uuid4)
    name: str = Field(..., description="Plugin name")
    version: str = Field(..., description="Semantic version")
    author: str = Field(..., description="Plugin author")
    description: str = Field(..., description="Plugin description")
    dependencies: list[str] = Field(
        default_factory=list, description="Required plugin names"
    )
    api_version: str = Field(default="1.0.0", description="Required API version")
    entry_point: str = Field(..., description="Main plugin class name")
    config: dict[str, Any] = Field(
        default_factory=dict, description="Plugin-specific configuration"
    )


class IPlugin(ABC):
    """
    Interface for all plugins.

    Plugins must implement this interface to be loaded by the plugin manager.
    """

    def __init__(self, metadata: PluginMetadata):
        """
        Initialize plugin.

        Args:
            metadata: Plugin metadata
        """
        self.metadata = metadata
        self.status = PluginStatus.UNLOADED
        self.loaded_at: datetime | None = None
        self.error: str | None = None

    @abstractmethod
    async def on_load(self) -> None:
        """
        Called when plugin is loaded.

        Use this to initialize resources, register handlers, etc.
        """

    @abstractmethod
    async def on_activate(self) -> None:
        """
        Called when plugin is activated.

        Use this to start background tasks, open connections, etc.
        """

    @abstractmethod
    async def on_deactivate(self) -> None:
        """
        Called when plugin is deactivated.

        Use this to stop background tasks, close connections, etc.
        """

    @abstractmethod
    async def on_unload(self) -> None:
        """
        Called when plugin is unloaded.

        Use this to cleanup resources, unregister handlers, etc.
        """

    @abstractmethod
    def get_info(self) -> dict[str, Any]:
        """
        Get plugin information.

        Returns:
            Dictionary with plugin information
        """
        return {
            "id": str(self.metadata.id),
            "name": self.metadata.name,
            "version": self.metadata.version,
            "author": self.metadata.author,
            "description": self.metadata.description,
            "status": self.status.value,
            "loaded_at": self.loaded_at.isoformat() if self.loaded_at else None,
            "error": self.error,
        }


class PluginContext:
    """
    Plugin execution context with dependency injection.

    Provides plugins with access to system resources and services.
    """

    def __init__(self):
        """Initialize plugin context."""
        self._services: dict[str, Any] = {}
        self._hooks: dict[str, list[Callable]] = {}

    def register_service(self, name: str, service: Any) -> None:
        """
        Register a service that plugins can use.

        Args:
            name: Service name
            service: Service instance
        """
        self._services[name] = service
        logger.info("Service registered for plugins", service=name)

    def get_service(self, name: str) -> Any:
        """
        Get a registered service.

        Args:
            name: Service name

        Returns:
            Service instance

        Raises:
            KeyError: If service not found
        """
        return self._services[name]

    def has_service(self, name: str) -> bool:
        """Check if service is registered."""
        return name in self._services

    def register_hook(self, hook_name: str, callback: Callable) -> None:
        """
        Register a hook callback.

        Args:
            hook_name: Hook identifier
            callback: Callback function
        """
        if hook_name not in self._hooks:
            self._hooks[hook_name] = []
        self._hooks[hook_name].append(callback)

    async def trigger_hook(self, hook_name: str, *args: Any, **kwargs: Any) -> list[Any]:
        """
        Trigger all callbacks for a hook.

        Args:
            hook_name: Hook identifier
            *args: Positional arguments for callbacks
            **kwargs: Keyword arguments for callbacks

        Returns:
            List of callback results
        """
        if hook_name not in self._hooks:
            return []

        results = []
        for callback in self._hooks[hook_name]:
            try:
                if inspect.iscoroutinefunction(callback):
                    result = await callback(*args, **kwargs)
                else:
                    result = callback(*args, **kwargs)
                results.append(result)
            except Exception as e:
                logger.error("Hook callback error", hook=hook_name, error=str(e))

        return results


class PluginManager:
    """
    Plugin manager with hot-loading and dependency injection.

    Manages plugin lifecycle, dependencies, and hot-reload without system restart.
    """

    def __init__(self, plugin_directory: Path | None = None):
        """
        Initialize plugin manager.

        Args:
            plugin_directory: Directory to scan for plugins
        """
        self.plugin_directory = plugin_directory or Path("plugins")
        self.plugins: dict[str, IPlugin] = {}
        self.plugin_modules: dict[str, Any] = {}
        self.context = PluginContext()
        logger.info("Plugin manager initialized", directory=str(self.plugin_directory))

    def register_service(self, name: str, service: Any) -> None:
        """
        Register a service for plugins to use.

        Args:
            name: Service name
            service: Service instance
        """
        self.context.register_service(name, service)

    async def load_plugin(self, plugin_path: Path) -> IPlugin | None:
        """
        Load a plugin from file.

        Args:
            plugin_path: Path to plugin file (.py)

        Returns:
            Loaded plugin instance or None if loading failed
        """
        try:
            # Load plugin metadata
            metadata_path = plugin_path.parent / f"{plugin_path.stem}_metadata.json"
            if not metadata_path.exists():
                logger.error("Plugin metadata not found", path=str(metadata_path))
                return None

            import json
            with open(metadata_path) as f:
                metadata_dict = json.load(f)
            metadata = PluginMetadata(**metadata_dict)

            # Check if already loaded
            if metadata.name in self.plugins:
                logger.warning("Plugin already loaded", plugin=metadata.name)
                return self.plugins[metadata.name]

            # Load plugin module
            spec = importlib.util.spec_from_file_location(metadata.name, plugin_path)
            if not spec or not spec.loader:
                logger.error("Failed to create module spec", path=str(plugin_path))
                return None

            module = importlib.util.module_from_spec(spec)
            self.plugin_modules[metadata.name] = module
            sys.modules[metadata.name] = module
            spec.loader.exec_module(module)

            # Get plugin class
            plugin_class = getattr(module, metadata.entry_point, None)
            if not plugin_class:
                logger.error(
                    "Plugin entry point not found",
                    plugin=metadata.name,
                    entry_point=metadata.entry_point,
                )
                return None

            # Instantiate plugin
            plugin = plugin_class(metadata)
            if not isinstance(plugin, IPlugin):
                logger.error("Plugin does not implement IPlugin", plugin=metadata.name)
                return None

            # Load plugin
            plugin.status = PluginStatus.LOADING
            await plugin.on_load()
            plugin.status = PluginStatus.LOADED
            plugin.loaded_at = datetime.utcnow()

            self.plugins[metadata.name] = plugin
            logger.info("Plugin loaded", plugin=metadata.name, version=metadata.version)

            return plugin

        except Exception as e:
            logger.error("Failed to load plugin", path=str(plugin_path), error=str(e))
            return None

    async def activate_plugin(self, plugin_name: str) -> bool:
        """
        Activate a loaded plugin.

        Args:
            plugin_name: Plugin name

        Returns:
            bool: True if activated successfully
        """
        if plugin_name not in self.plugins:
            logger.error("Plugin not found", plugin=plugin_name)
            return False

        plugin = self.plugins[plugin_name]

        if plugin.status == PluginStatus.ACTIVE:
            logger.warning("Plugin already active", plugin=plugin_name)
            return True

        try:
            await plugin.on_activate()
            plugin.status = PluginStatus.ACTIVE
            logger.info("Plugin activated", plugin=plugin_name)
            return True

        except Exception as e:
            plugin.status = PluginStatus.ERROR
            plugin.error = str(e)
            logger.error("Failed to activate plugin", plugin=plugin_name, error=str(e))
            return False

    async def deactivate_plugin(self, plugin_name: str) -> bool:
        """
        Deactivate an active plugin.

        Args:
            plugin_name: Plugin name

        Returns:
            bool: True if deactivated successfully
        """
        if plugin_name not in self.plugins:
            logger.error("Plugin not found", plugin=plugin_name)
            return False

        plugin = self.plugins[plugin_name]

        if plugin.status != PluginStatus.ACTIVE:
            logger.warning("Plugin not active", plugin=plugin_name)
            return True

        try:
            await plugin.on_deactivate()
            plugin.status = PluginStatus.LOADED
            logger.info("Plugin deactivated", plugin=plugin_name)
            return True

        except Exception as e:
            plugin.status = PluginStatus.ERROR
            plugin.error = str(e)
            logger.error("Failed to deactivate plugin", plugin=plugin_name, error=str(e))
            return False

    async def unload_plugin(self, plugin_name: str) -> bool:
        """
        Unload a plugin (hot-unload).

        Args:
            plugin_name: Plugin name

        Returns:
            bool: True if unloaded successfully
        """
        if plugin_name not in self.plugins:
            logger.error("Plugin not found", plugin=plugin_name)
            return False

        plugin = self.plugins[plugin_name]

        try:
            # Deactivate if active
            if plugin.status == PluginStatus.ACTIVE:
                await self.deactivate_plugin(plugin_name)

            # Unload
            plugin.status = PluginStatus.UNLOADING
            await plugin.on_unload()

            # Remove from system
            del self.plugins[plugin_name]
            if plugin_name in self.plugin_modules:
                del sys.modules[plugin_name]
                del self.plugin_modules[plugin_name]

            logger.info("Plugin unloaded", plugin=plugin_name)
            return True

        except Exception as e:
            plugin.status = PluginStatus.ERROR
            plugin.error = str(e)
            logger.error("Failed to unload plugin", plugin=plugin_name, error=str(e))
            return False

    async def reload_plugin(self, plugin_name: str) -> bool:
        """
        Hot-reload a plugin without system restart.

        Args:
            plugin_name: Plugin name

        Returns:
            bool: True if reloaded successfully
        """
        if plugin_name not in self.plugins:
            logger.error("Plugin not found", plugin=plugin_name)
            return False

        plugin = self.plugins[plugin_name]
        plugin_path = self.plugin_directory / f"{plugin_name}.py"

        logger.info("Reloading plugin", plugin=plugin_name)

        # Unload current version
        was_active = plugin.status == PluginStatus.ACTIVE
        if not await self.unload_plugin(plugin_name):
            return False

        # Load new version
        new_plugin = await self.load_plugin(plugin_path)
        if not new_plugin:
            return False

        # Activate if it was active before
        if was_active:
            return await self.activate_plugin(plugin_name)

        return True

    async def discover_and_load_all(self) -> list[IPlugin]:
        """
        Discover and load all plugins in plugin directory.

        Returns:
            List of loaded plugins
        """
        if not self.plugin_directory.exists():
            logger.warning("Plugin directory not found", dir=str(self.plugin_directory))
            return []

        loaded_plugins: list[IPlugin] = []

        for plugin_file in self.plugin_directory.glob("*.py"):
            if plugin_file.stem.endswith("_metadata"):
                continue

            plugin = await self.load_plugin(plugin_file)
            if plugin:
                loaded_plugins.append(plugin)

        logger.info("Plugin discovery complete", count=len(loaded_plugins))
        return loaded_plugins

    def get_plugin(self, plugin_name: str) -> IPlugin | None:
        """Get a loaded plugin by name."""
        return self.plugins.get(plugin_name)

    def list_plugins(self) -> list[dict[str, Any]]:
        """
        List all loaded plugins with their info.

        Returns:
            List of plugin info dictionaries
        """
        return [plugin.get_info() for plugin in self.plugins.values()]

    def get_active_plugins(self) -> list[IPlugin]:
        """Get all active plugins."""
        return [p for p in self.plugins.values() if p.status == PluginStatus.ACTIVE]


# Example plugin implementation
class ExamplePlugin(IPlugin):
    """Example plugin demonstrating the plugin interface."""

    async def on_load(self) -> None:
        """Load plugin resources."""
        logger.info("Example plugin loading", plugin=self.metadata.name)

    async def on_activate(self) -> None:
        """Activate plugin."""
        logger.info("Example plugin activated", plugin=self.metadata.name)

    async def on_deactivate(self) -> None:
        """Deactivate plugin."""
        logger.info("Example plugin deactivated", plugin=self.metadata.name)

    async def on_unload(self) -> None:
        """Unload plugin resources."""
        logger.info("Example plugin unloading", plugin=self.metadata.name)

    def get_info(self) -> dict[str, Any]:
        """Get plugin info."""
        return {
            **super().get_info(),
            "custom_field": "example_value",
        }

