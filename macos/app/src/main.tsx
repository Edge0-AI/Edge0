import React from "react";
import ReactDOM from "react-dom/client";
import { RouterProvider } from "@tanstack/react-router";

import "./i18n";
import { applyFont, applyTheme, storedFont, storedTheme } from "./lib/theme";
import { router } from "./router";
import "./styles.css";
import "katex/dist/katex.min.css";

applyTheme(storedTheme());
applyFont(storedFont());

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <RouterProvider router={router} />
  </React.StrictMode>,
);
