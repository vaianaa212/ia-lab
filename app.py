import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.cases import get_formulations
from src.formulation import to_export_dict
from src.search import bfs, dfs, ucs, astar, manhattan
from src.metaheuristics import hill_climb, simulated_annealing
from src.ml import train_classifier, train_regressor, compute_confusion, tune_classifier
from src.monitoring import drift_report

st.set_page_config(page_title="AI for Business Analytics — Decision App", layout="wide")

st.title("AI for Business Analytics — Decision Support App")
st.caption("End-to-end app: formulation → search/optimization → ML → interpretability/monitoring.")

with st.sidebar:
    st.header("Módulo")
    module = st.radio(
        "Selecciona",
        ["1) Formulación (S,A,T,G,C,R)",
         "2) Búsqueda (BFS/DFS/UCS/A*)",
         "3) Metaheurísticas (visión)",
         "4) ML supervisado (pipeline)",
         "5) Evaluación, tuning y riesgos",
         "6) Resumen del curso"]
    )

DATA_DIR = "data"

def load_csv(name):
    return pd.read_csv(f"{DATA_DIR}/{name}")

def section_formulation():
    st.subheader("1) Formulación del problema")
    forms = get_formulations()
    choice = st.selectbox("Caso/representación", list(forms.keys()))
    f = forms[choice]
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Plantilla")
        st.write("**Estado (schema)**", f.state_schema)
        st.write("**Acción (schema)**", f.action_schema)
        st.write("**Transición**", f.transition_desc)
        st.write("**Objetivo**", f.goal_test_desc)
        st.write("**Coste**", f.cost_desc)
    with col2:
        st.markdown("### Restricciones")
        st.write("**Hard (duras)**")
        st.write(f.hard_constraints)
        st.write("**Soft (blandas / preferencias)**")
        st.write(f.soft_constraints)
        st.markdown("### Ejemplos")
        st.json(f.examples)

    export = to_export_dict(f)
    st.download_button("Descargar formulación (JSON)", data=pd.Series(export).to_json(), file_name=f"{f.name}.json")

def draw_grid(grid, path=None, start=None, goal=None):
    arr = np.array(grid)
    fig = plt.figure()
    plt.imshow(arr)
    if path:
        xs = [p[1] for p in path]
        ys = [p[0] for p in path]
        plt.plot(xs, ys)
    if start:
        plt.scatter([start[1]], [start[0]])
    if goal:
        plt.scatter([goal[1]], [goal[0]])
    plt.xticks([]); plt.yticks([])
    st.pyplot(fig)

def section_search():
    st.subheader("2) Búsqueda en un grid (picking / rutas internas)")
    st.write("Demostración conceptual de BFS/DFS/UCS/A* en un mapa de almacén.")
    n = st.slider("Tamaño del grid", 8, 25, 14)
    obstacle_rate = st.slider("Densidad de obstáculos", 0.0, 0.35, 0.18)
    rng = np.random.default_rng(7)
    grid = (rng.random((n,n)) < obstacle_rate).astype(int).tolist()
    # ensure borders free a bit
    for i in range(n):
        grid[0][i]=0; grid[n-1][i]=0; grid[i][0]=0; grid[i][n-1]=0
    start = (0,0)
    goal = (n-1,n-1)
    grid[start[0]][start[1]] = 0
    grid[goal[0]][goal[1]] = 0

    algo = st.selectbox("Algoritmo", ["BFS", "DFS", "UCS", "A* (Manhattan)"])
    if st.button("Ejecutar"):
        if algo == "BFS":
            res = bfs(start, goal, grid)
        elif algo == "DFS":
            res = dfs(start, goal, grid)
        elif algo == "UCS":
            res = ucs(start, goal, grid)
        else:
            res = astar(start, goal, grid, h=manhattan)

        st.write({"found": res.found, "explored": res.explored, "cost": res.cost, "path_len": len(res.path)})
        draw_grid(grid, res.path if res.found else None, start, goal)

    st.info("Idea clave: la representación (estado/acciones/coste) determina el tamaño del espacio; el algoritmo explora ese espacio.")

def section_metaheuristics():
    st.subheader("3) Metaheurísticas (visión práctica)")
    st.write("Ejemplo: minimizar una función con óptimos locales (analogía con optimización combinatoria).")
    # Define a bumpy function
    def f(x,y):
        return (x**2 + y**2) + 3*np.sin(x) + 2*np.cos(y)

    x0 = st.slider("x0", -10.0, 10.0, 4.0)
    y0 = st.slider("y0", -10.0, 10.0, -3.0)

    col1,col2 = st.columns(2)
    with col1:
        st.markdown("### Hill Climbing")
        step = st.slider("step", 0.1, 2.0, 0.7)
        iters = st.slider("iters", 50, 2000, 400)
        if st.button("Run Hill Climb"):
            r = hill_climb(f, x0, y0, step=step, iters=iters)
            st.write(r)

    with col2:
        st.markdown("### Simulated Annealing")
        t0 = st.slider("t0", 0.5, 10.0, 5.0)
        alpha = st.slider("alpha", 0.90, 0.999, 0.985)
        iters2 = st.slider("iters (SA)", 100, 4000, 1200)
        if st.button("Run Simulated Annealing"):
            r = simulated_annealing(f, x0, y0, t0=t0, alpha=alpha, iters=iters2)
            st.write(r)

    st.caption("Mensaje: en problemas reales, aceptamos aproximaciones cuando la búsqueda exacta es inviable.")

def section_ml():
    st.subheader("4) ML supervisado (pipeline)")
    task = st.selectbox("Tarea", ["Churn (clasificación)", "Demanda (regresión)", "Fraude (clasificación desbalanceada)"])

    if task.startswith("Churn"):
        df = load_csv("churn.csv")
        target = "churn"
        model = st.selectbox("Modelo", ["logreg","tree","rf","gb","mlp"])
        pipe, rep, extra = train_classifier(df, target, model)
        st.write(rep.metrics)
        cm = compute_confusion(extra["y_true"], extra["y_pred"])
        st.write("Matriz de confusión", cm)
        st.caption(rep.notes)

    elif task.startswith("Demanda"):
        df = load_csv("demand.csv")
        target = "demand"
        model = st.selectbox("Modelo", ["linreg","ridge","lasso","tree","rf","gb","mlp"])
        pipe, rep, extra = train_regressor(df, target, model)
        st.write(rep.metrics)
        st.caption(rep.notes)

    else:
        df = load_csv("fraud.csv")
        target = "fraud"
        model = st.selectbox("Modelo", ["logreg","tree","rf","gb","mlp"])
        pipe, rep, extra = train_classifier(df, target, model)
        st.write(rep.metrics)
        cm = compute_confusion(extra["y_true"], extra["y_pred"])
        st.write("Matriz de confusión", cm)
        st.warning("Dataset desbalanceado: accuracy puede engañar; mira recall/precision/F1 y coste de error.")
        st.caption(rep.notes)

def section_eval_risks():
    st.subheader("5) Evaluación, tuning y riesgos")
    st.write("Tuning (grid search) + monitoreo de deriva (PSI) sobre un ejemplo.")
    df = load_csv("churn.csv")
    target = "churn"
    model = st.selectbox("Tuning modelo", ["logreg","rf","tree"])
    if st.button("Ejecutar GridSearchCV (puede tardar)"):
        gs = tune_classifier(df, target, model)
        st.write("Mejor score CV:", float(gs.best_score_))
        st.write("Mejores params:", gs.best_params_)

    st.markdown("### Deriva (drift) — ejemplo PSI")
    st.write("Simulamos un 'nuevo mes' con cambios en distribución (más tickets, más cargos).")
    df_new = df.copy()
    df_new["support_tickets_3m"] = (df_new["support_tickets_3m"] + 1).clip(0, None)
    df_new["monthly_charges"] = df_new["monthly_charges"] * 1.05
    numeric_cols = ["tenure_months","monthly_charges","support_tickets_3m","usage_gb"]
    report = drift_report(df[numeric_cols], df_new[numeric_cols], numeric_cols)
    st.write(report)
    st.info("Interpretación aproximada: PSI cercano a 0 = sin deriva; >0.2 = atención; >0.3 = significativa.")

def section_summary():
    st.subheader("6) Resumen del curso (mapa conceptual)")
    st.markdown("""
- **Formulación (Sesión 2):** (S,A,T,G,C,R) + hard/soft + trade-offs.
- **Búsqueda (Sesiones 5–8):** BFS/DFS/UCS/A* + heurísticas (admisibilidad/consistencia).
- **Optimización (Sesiones 9–10):** metaheurísticas + pipeline de decisión del caso al algoritmo.
- **ML (Sesiones 11–20):** workflow, overfitting, métricas, modelos lineales, árboles, ensambles, features, tuning.
- **Interpretabilidad y riesgos (Sesiones 21–22):** explicabilidad, fairness, desbalance, drift.
- **Redes neuronales (Sesiones 23–26):** MLP, entrenamiento, regularización, comparación por tipo de datos.
- **Buenas prácticas (Sesiones 27–30):** reproducibilidad, trazabilidad, comunicación, proyecto final.
""")
    st.success("Siguiente paso natural: añadir el módulo de 'búsqueda aplicada a negocio' usando la misma formulación (actividad casa 1).")

if module.startswith("1)"):
    section_formulation()
elif module.startswith("2)"):
    section_search()
elif module.startswith("3)"):
    section_metaheuristics()
elif module.startswith("4)"):
    section_ml()
elif module.startswith("5)"):
    section_eval_risks()
else:
    section_summary()
