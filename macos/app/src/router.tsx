import { createRootRoute, createRoute, createRouter, redirect } from "@tanstack/react-router";

import { AppShell } from "./components/AppShell";
import { ChatPage } from "./routes/chat";
import { ModelsPage } from "./routes/models";
import { ServicePage } from "./routes/service";
import { SettingsPage } from "./routes/settings";

const rootRoute = createRootRoute({
  component: AppShell,
});

const chatRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/",
  component: ChatPage,
});

const modelsRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/models",
  component: ModelsPage,
});

const settingsRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/settings",
  component: SettingsPage,
});

const serviceRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/service",
  component: ServicePage,
});

const legacyChatRedirect = createRoute({
  getParentRoute: () => rootRoute,
  path: "/chat",
  beforeLoad: () => {
    throw redirect({ to: "/" });
  },
});

const routeTree = rootRoute.addChildren([
  chatRoute,
  modelsRoute,
  legacyChatRedirect,
  settingsRoute,
  serviceRoute,
]);

export const router = createRouter({ routeTree });

declare module "@tanstack/react-router" {
  interface Register {
    router: typeof router;
  }
}
