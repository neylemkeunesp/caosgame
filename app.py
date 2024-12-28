import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# Set dark mode for matplotlib
plt.style.use('dark_background')

# Custom dark theme for Plotly
plotly_dark_theme = dict(
    layout=dict(
        plot_bgcolor='#0E1117',
        paper_bgcolor='#0E1117',
        font=dict(color='white'),
        xaxis=dict(
            gridcolor='rgba(128, 128, 128, 0.2)',
            zerolinecolor='rgba(128, 128, 128, 0.2)'
        ),
        yaxis=dict(
            gridcolor='rgba(128, 128, 128, 0.2)',
            zerolinecolor='rgba(128, 128, 128, 0.2)'
        )
    )
)

class ChaoticSystem:
    def __init__(self, r):
        self.r = r
    
    def logistic_map(self, x):
        return self.r * x * (1 - x)
    
    def iterate(self, x0, n):
        trajectory = np.zeros(n)
        x = x0
        for i in range(n):
            trajectory[i] = x
            x = self.logistic_map(x)
        return trajectory
    
    def lyapunov(self, x0, n):
        x = x0
        lyap = 0
        for i in range(n):
            x = self.logistic_map(x)
            # Derivative of logistic map: r(1-2x)
            lyap += np.log(abs(self.r * (1 - 2*x)))
        return lyap / n
    
    def bifurcation_data(self, r_range, n_r, n_iterations, n_discard):
        r_values = np.linspace(r_range[0], r_range[1], n_r)
        x_values = []
        r_plot = []
        
        for r in r_values:
            self.r = r
            x = 0.5  # Initial condition
            
            # Discard transients
            for _ in range(n_discard):
                x = self.logistic_map(x)
            
            # Collect points for bifurcation diagram
            for _ in range(n_iterations):
                x = self.logistic_map(x)
                x_values.append(x)
                r_plot.append(r)
                
        return np.array(r_plot), np.array(x_values)

def plot_trajectory(r, x0, n):
    # Create a new system instance with the current r value
    system = ChaoticSystem(r)
    trajectory = system.iterate(x0, n)
    
    fig, ax = plt.subplots(figsize=(8, 4), facecolor='#0E1117')
    
    # Plot trajectory
    ax.plot(range(n), trajectory, '#00ff00', label='Trajectory', linewidth=1.5)
    
    ax.set_xlabel('Iteration', fontsize=12, color='white')
    ax.set_ylabel('x', fontsize=12, color='white')
    ax.set_title(f'System Evolution (r = {r:.3f})', fontsize=14, pad=20, color='white')
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=10)
    ax.set_facecolor('#0E1117')
    for spine in ax.spines.values():
        spine.set_color('white')
    ax.tick_params(colors='white')
    return fig

def plot_lyapunov_vs_r(system, r_range, n_r=500):
    r_values = np.linspace(r_range[0], r_range[1], n_r)
    lyap_values = np.zeros(n_r)
    
    for i, r in enumerate(r_values):
        system.r = r
        lyap_values[i] = system.lyapunov(0.5, 100)  # Use x0=0.5 and 100 iterations
    
    fig, ax = plt.subplots(figsize=(10, 4), facecolor='#0E1117')
    ax.plot(r_values, lyap_values, '#00ff00', linewidth=1)
    ax.axhline(y=0, color='#ff6b6b', linestyle='--', alpha=0.5)
    
    ax.set_xlabel('r parameter', fontsize=12, color='white')
    ax.set_ylabel('Lyapunov Exponent', fontsize=12, color='white')
    ax.set_title('Lyapunov Exponent vs r', fontsize=14, pad=20, color='white')
    ax.grid(True, alpha=0.2)
    ax.set_facecolor('#0E1117')
    for spine in ax.spines.values():
        spine.set_color('white')
    ax.tick_params(colors='white')
    return fig

def plot_bifurcation(system, r_range, n_r=1000, n_iterations=100, n_discard=100):
    r_plot, x_values = system.bifurcation_data(r_range, n_r, n_iterations, n_discard)
    
    # Calculate Lyapunov exponent for each r value
    lyap_values = np.zeros(n_r)
    unique_r_values = np.unique(r_plot)
    for i, r in enumerate(unique_r_values):
        system.r = r
        lyap_values[i] = system.lyapunov(0.5, 100)
    
    # Create interpolation function for Lyapunov values
    from scipy.interpolate import interp1d
    lyap_interp = interp1d(unique_r_values, lyap_values, kind='linear', fill_value='extrapolate')
    
    # Get Lyapunov values for all r points
    all_lyap_values = lyap_interp(r_plot)
    
    # Create Plotly figure
    fig = go.Figure()
    
    # Add scatter plot for bifurcation points
    fig.add_trace(go.Scattergl(
        x=r_plot,
        y=x_values,
        mode='markers',
        marker=dict(
            color='#00ff00',
            size=1,
            opacity=0.1
        ),
        hovertemplate=(
            'r: %{x:.3f}<br>' +
            'x: %{y:.3f}<br>' +
            'λ: %{customdata:.3f}<br>' +
            '<extra></extra>'
        ),
        customdata=all_lyap_values
    ))
    
    # Add vertical lines for critical points
    critical_points = {
        1: 'Stable fixed point',
        3: 'First bifurcation',
        3.57: 'Onset of chaos'
    }
    
    for r, label in critical_points.items():
        if r_range[0] <= r <= r_range[1]:
            fig.add_vline(x=r, line_color='#ff6b6b', line_dash='dash', line_width=1, opacity=0.3)
            fig.add_annotation(x=r, y=-0.05, text=f'r={r}', showarrow=False, 
                             textangle=-90, font=dict(color='white', size=8))
    
    # Update layout with dark theme
    fig.update_layout(
        **plotly_dark_theme['layout'],
        showlegend=False,
        title=dict(
            text='Bifurcation Diagram',
            font=dict(size=14),
            y=0.95
        ),
        xaxis_title='r parameter',
        yaxis_title='x value',
        hovermode='closest',
        width=1000,
        height=250,
        margin=dict(t=30, b=30, l=50, r=20)
    )
    
    return fig

# Set Streamlit theme to dark
st.set_page_config(
    page_title="1D Chaotic Systems",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "A visualization tool for 1D chaotic systems"
    }
)

# Force dark theme
st.markdown("""
    <style>
        .stApp {
            background-color: #0E1117;
            color: white;
        }
        .stMarkdown {
            color: white;
        }
        .stSidebar {
            background-color: #262730;
        }
        .stSlider {
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar controls
st.sidebar.title('1D Chaotic Systems Visualizer')
st.sidebar.write('Explore the behavior of the logistic map: x_{n+1} = rx_n(1-x_n)')
st.sidebar.header('Parameters')
r = st.sidebar.slider('r parameter', 0.0, 4.0, 3.7, 0.01)
x0 = st.sidebar.slider('Initial condition (x0)', 0.0, 1.0, 0.5, 0.01)
n_iterations = st.sidebar.slider('Number of iterations', 10, 1000, 100)
show_cobweb = st.sidebar.checkbox('Show cobweb diagram', value=False)

# Advanced settings
with st.sidebar.expander("Advanced Settings"):
    bifurcation_points = st.slider('Bifurcation points', 100, 5000, 1000)
    iterations_per_point = st.slider('Iterations per point', 10, 200, 100)

# Create system instance
system = ChaoticSystem(r)

def plot_cobweb(r, x0, n):
    # Create a new system instance with the current r value
    system = ChaoticSystem(r)
    fig, ax = plt.subplots(figsize=(8, 8), facecolor='#0E1117')
    
    # Plot the functions
    x_range = np.linspace(0, 1, 1000)
    y_logistic = r * x_range * (1 - x_range)
    ax.plot(x_range, y_logistic, '#ff6b6b', label='f(x) = rx(1-x)', alpha=0.8, linewidth=2)
    ax.plot(x_range, x_range, '#4dabf7', label='y = x', alpha=0.8, linewidth=2)
    
    # Skip first 200 points (transients)
    x = x0
    for _ in range(200):
        x = system.logistic_map(x)
    
    # Store points to check for returns
    points = []
    max_points = 1000  # Safety limit
    
    # Draw cobweb until return or max points reached
    while len(points) < max_points:
        y = system.logistic_map(x)
        
        # Check if we've returned close to a previous point
        for prev_x, prev_y in points:
            if abs(x - prev_x) < 1e-5 and abs(y - prev_y) < 1e-5:
                # Draw final connecting lines
                ax.plot([x, x], [x, y], '#ffd43b', linewidth=1)
                ax.plot([x, y], [y, y], '#ffd43b', linewidth=1)
                return fig
        
        # Draw current lines
        ax.plot([x, x], [x, y], '#ffd43b', linewidth=1)
        ax.plot([x, y], [y, y], '#ffd43b', linewidth=1)
        
        # Store point and continue
        points.append((x, y))
        x = y
    
    ax.set_xlabel('x', fontsize=12, color='white')
    ax.set_ylabel('y', fontsize=12, color='white')
    ax.set_title(f'Cobweb Diagram (r = {r:.3f})', fontsize=14, pad=20, color='white')
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=10)
    ax.set_facecolor('#0E1117')
    for spine in ax.spines.values():
        spine.set_color('white')
    ax.tick_params(colors='white')
    return fig

# Main content
st.markdown("""
<style>
    div.block-container{padding-top: 0.5rem;}
    div.stMarkdown {margin-bottom: 0.5rem;}
    div.element-container {margin-bottom: 0.5rem;}
</style>
""", unsafe_allow_html=True)

# Bifurcation diagram at the very top
st.markdown('<p style="margin: 0 0 0.2rem 0"><strong>Bifurcation Diagram</strong> (Hover over points to see r value and Lyapunov exponent)</p>', unsafe_allow_html=True)
r_range = st.slider('r range for bifurcation', 0.0, 4.0, (2.5, 4.0))
fig_bif = plot_bifurcation(system, r_range, n_r=bifurcation_points, n_iterations=iterations_per_point)
st.plotly_chart(fig_bif, use_container_width=True)

# Two-column layout for trajectory and cobweb
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown('<p style="margin: 0 0 0.2rem 0"><strong>System Trajectory</strong></p>', unsafe_allow_html=True)
    fig_traj = plot_trajectory(r, x0, n_iterations)
    st.pyplot(fig_traj)

with col2:
    st.markdown('<p style="margin: 0 0 0.2rem 0"><strong>Cobweb Diagram</strong></p>', unsafe_allow_html=True)
    if show_cobweb:
        fig_cobweb = plot_cobweb(r, x0, n_iterations)
        st.pyplot(fig_cobweb)

# Lyapunov information and plot in two columns
col3, col4 = st.columns([1, 2])

with col3:
    st.markdown('<p style="margin: 0 0 0.2rem 0"><strong>Lyapunov Exponent</strong></p>', unsafe_allow_html=True)
    lyap = system.lyapunov(x0, n_iterations)
    
    # Style the Lyapunov number with dark theme colors
    if lyap > 0:
        st.markdown(f'<h3 style="color: #ff6b6b;">λ = {lyap:.4f}</h3>', unsafe_allow_html=True)
        st.write('System is **chaotic** (λ > 0)')
    elif lyap < 0:
        st.markdown(f'<h3 style="color: #69db7c;">λ = {lyap:.4f}</h3>', unsafe_allow_html=True)
        st.write('System is **stable** (λ < 0)')
    else:
        st.markdown(f'<h3 style="color: #ffd43b;">λ = {lyap:.4f}</h3>', unsafe_allow_html=True)
        st.write('System is at **bifurcation point** (λ ≈ 0)')

with col4:
    st.markdown('<p style="margin: 0 0 0.2rem 0"><strong>Lyapunov Exponent vs r</strong></p>', unsafe_allow_html=True)
    fig_lyap = plot_lyapunov_vs_r(system, r_range)
    st.pyplot(fig_lyap)

# Information section
with st.expander("About the Visualizations"):
    st.markdown("""
    ### Key Components
    
    1. **Trajectory Plot**
        - Shows how the system evolves over time
        - Optional cobweb diagram visualizes the iteration process
        - Green line shows the actual trajectory
        
    2. **Lyapunov Exponent (λ)**
        - Measures sensitivity to initial conditions
        - λ > 0: Chaotic behavior (exponential divergence)
        - λ < 0: Stable behavior (convergence)
        - λ ≈ 0: Bifurcation point
        
    3. **Bifurcation Diagram**
        - Shows long-term behavior as r varies
        - Vertical lines mark critical points
        - Period doubling route to chaos
        - Notable features:
            - r < 3: Single stable point
            - r ≈ 3: First bifurcation
            - r ≈ 3.57: Onset of chaos
            - Beyond 3.57: Chaos with periodic windows
            
    4. **Lyapunov Exponent vs r**
        - Shows how the system's chaotic behavior changes with r
        - Negative values indicate stable behavior
        - Positive values indicate chaos
        - Zero crossings correspond to bifurcation points
    """)
