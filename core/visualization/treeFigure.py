import html
import math

import plotly.graph_objects as go

INTERNAL = {
    'face': '#0026b6',
    'edge': '#000000',
    'text': '#ffffff',
    'width': 2
}

LEAF = {
    'face': '#1a7300',
    'edge': '#000000',
    'text': '#ffffff',
    'width': 2
}

CONFIG = {
    'scrollZoom': True,
    'displaylogo': False,
    'responsive': True,
    'doubleClick': 'reset',
    'modeBarButtonsToRemove': ['select2d', 'lasso2d'],
}


def esc(text):
    return html.escape(str(text), quote=False)


def format_text(title, lines):
    parts = ['<b>' + esc(title) + '</b>']
    for line in lines:
        parts.append(esc(line))
    return '<br>'.join(parts)


def place(nodes, node_id, depth, state, positions):
    node = nodes[node_id]
    if len(node['children']) == 0:
        x = 16 + state['slot'] * (150 + 18) + 150 / 2.0
        state['slot'] = state['slot'] + 1
    else:
        centers = []
        for child in node['children']:
            place(nodes, child['target'], depth + 1, state, positions)
            centers.append(positions[child['target']]['x'])
        x = (centers[0] + centers[len(centers) - 1]) / 2.0
    positions[node_id] = {'x': x, 'depth': depth}
    if depth > state['depth']:
        state['depth'] = depth


def center_y(depth, height):
    half = max(58, 100) / 2.0
    return height - 16 - half - depth * 128


def half_height(node):
    if len(node['children']) == 0:
        return 70 / 2.0
    return 58 / 2.0


def rectangle_points(cx, cy):
    x0 = cx - 150 / 2.0
    x1 = cx + 150 / 2.0
    y0 = cy - 58 / 2.0
    y1 = cy + 58 / 2.0
    return [x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0]


def ellipse_points(cx, cy):
    xs = []
    ys = []
    radius_x = 130 / 2.0
    radius_y = 70 / 2.0

    for index in range(72 + 1):
        angle = 2 * math.pi * index / 72
        xs.append(cx + radius_x * math.cos(angle))
        ys.append(cy + radius_y * math.sin(angle))

    return xs, ys


def label_position(start_x, start_y, end_x, end_y, is_left):
    dx = end_x - start_x
    dy = end_y - start_y
    length = math.hypot(dx, dy)
    normal_x = -dy / length
    normal_y = dx / length
    if is_left and normal_x > 0:
        normal_x = -normal_x
        normal_y = -normal_y
    if not is_left and normal_x < 0:
        normal_x = -normal_x
        normal_y = -normal_y
    middle_x = (start_x + end_x) / 2.0
    middle_y = (start_y + end_y) / 2.0

    return middle_x + normal_x * 17, middle_y + normal_y * 17


def build_figure(nodes, positions, width, height):
    figure = go.Figure()
    edge_x = []
    edge_y = []
    edge_labels = []

    for node_id in nodes:
        node = nodes[node_id]

        if len(node['children']) == 0:
            continue

        start_x = positions[node_id]['x']
        start_y = center_y(positions[node_id]['depth'], height) - half_height(node)

        for child in node['children']:
            target = child['target']

            end_x = positions[target]['x']
            end_y = center_y(positions[target]['depth'], height) + half_height(nodes[target])

            edge_x.extend([start_x, end_x, None])
            edge_y.extend([start_y, end_y, None])

            label_x, label_y = label_position(start_x, start_y, end_x, end_y, child['label'] == 'tak')

            edge_labels.append({'x': label_x, 'y': label_y, 'text': child['label']})

    figure.add_trace(go.Scatter(
        x=edge_x,
        y=edge_y,
        mode='lines',
        line={
            'color': '#111111',
            'width': 2
        },
        hoverinfo='skip',
        showlegend=False,
    ))

    for node_id in nodes:
        node = nodes[node_id]

        cx = positions[node_id]['x']
        cy = center_y(positions[node_id]['depth'], height)

        if node['leaf']:
            style = LEAF
            xs, ys = ellipse_points(cx, cy)
        else:
            style = INTERNAL
            xs, ys = rectangle_points(cx, cy)

        figure.add_trace(go.Scatter(
            x=xs,
            y=ys,
            mode='lines',
            fill='toself',
            fillcolor=style['face'],
            line={
                'color': style['edge'],
                'width': style['width']
            },
            hoveron='fills',
            text=format_text(node['hover_title'], node['hover_lines']),
            hoverinfo='text',
            showlegend=False,
        ))

        figure.add_annotation(
            x=cx,
            y=cy,
            text=format_text(node['title'], node['lines']),
            showarrow=False,
            font={
                'size': 11,
                'color': style['text']
            },
        )

    for label in edge_labels:
        figure.add_annotation(
            x=label['x'],
            y=label['y'],
            text=label['text'],
            showarrow=False,
            font={
                'size': 12,
                'color': '#444444'
            },
        )

    figure.add_trace(go.Scatter(
        x=[None],
        y=[None],
        mode='markers',
        name='Węzeł decyzyjny',
        marker={
            'symbol': 'square',
            'size': 14,
            'color': INTERNAL['face'],
            'line': {
                'color': INTERNAL['edge'],
                'width': 2
            }
        },
    ))

    figure.add_trace(go.Scatter(
        x=[None],
        y=[None],
        mode='markers',
        name='Liść',
        marker={
            'symbol': 'circle',
            'size': 14,
            'color': LEAF['face'],
            'line': {
                'color': LEAF['edge'],
                'width': 2
            }
        },
    ))

    figure.add_trace(go.Scatter(
        x=[None],
        y=[None],
        mode='markers',
        name='Krawędzie',
        marker={
            'symbol': 'line-nw-open',
            'size': 14,
            'color': '#111111'
        },
    ))

    figure.update_layout(
        autosize=True,
        margin={'l': 0, 'r': 0, 't': 0, 'b': 0},
        paper_bgcolor='pink',
        plot_bgcolor='white',
        showlegend=True,
        legend={
            'orientation': 'h',
            'x': 0.01,
            'y': 0.99,
            'xanchor': 'left',
            'yanchor': 'top',
            'bgcolor': 'rgba(255,255,255,0.92)',
            'bordercolor': '#d9d9d9',
            'borderwidth': 1,
            'itemclick': False,
            'itemdoubleclick': False,
            'font': {
                'size': 12
            },
        },
        hovermode='closest',
        dragmode='pan',
        hoverlabel={
            'align': 'left',
            'font':
                {
                    'size': 13
                }
        },
        xaxis={
            'visible': False,
            'range':[0, width]
        },
        yaxis={
            'visible': False,
            'range':[0, height],
            'scaleanchor': 'x',
            'scaleratio': 1
        },
    )

    return figure


def build_tree_html(nodes):
    state = {'slot': 0, 'depth': 0}
    positions = {}
    place(nodes, 0, 0, state, positions)

    levels = state['depth'] + 1
    width = state['slot'] * (150 + 18) - 18 + 2 * 16
    height = (levels - 1) * 128 + max(58, 70) + 2 * 16
    figure = build_figure(nodes, positions, width, height)

    return figure.to_html(
        full_html=False,
        include_plotlyjs=True,
        config=CONFIG,
        default_width='100%',
        default_height='100%',
        div_id='drzewo',
    )
