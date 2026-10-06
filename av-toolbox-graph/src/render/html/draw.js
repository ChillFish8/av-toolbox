(function () {
    "use strict";

    const SVG_NS = "http://www.w3.org/2000/svg";
    const FONT_FAMILY = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace";
    const FONT = "13px " + FONT_FAMILY;
    const HEADING_FONT = "12px " + FONT_FAMILY;
    const HEADING_INSET = 12;
    const FRAME_PADDING = 16;
    const LINE_HEIGHT = 18;
    const PADDING_X = 12;
    const PADDING_Y = 8;
    const SHADOW = 6;
    const EXCHANGE_SIZE = 44;
    const LABEL_HEIGHT = 14;

    const diagram = JSON.parse(document.getElementById("diagram").textContent);
    const canvas = document.getElementById("canvas");
    const message = document.getElementById("message");

    function showMessage(content) {
        message.textContent = content;
        message.hidden = false;
    }

    if (diagram.boxes.length === 0) {
        showMessage("This graph has no stages to draw.");
        return;
    }

    if (typeof ELK === "undefined") {
        showMessage("The layout library could not be loaded, so the graph cannot be drawn. Check that cdn.jsdelivr.net is reachable.");
        return;
    }

    const measure = document.createElement("canvas").getContext("2d");
    measure.font = FONT;

    const boxesById = new Map(diagram.boxes.map(box => [box.id, box]));
    const exchangesById = new Map(diagram.exchanges.map(exchange => [exchange.id, exchange]));
    const blocksById = new Map(diagram.blocks.map(block => [block.id, block]));
    const blockOrigins = new Map();
    const boxGroups = new Map();
    const linkLines = new Map();

    function textWidth(content) {
        return measure.measureText(content).width;
    }

    function headingWidth(content) {
        measure.font = HEADING_FONT;
        const width = measure.measureText(content).width;
        measure.font = FONT;
        return width;
    }

    function frameOptions(block) {
        const minimumWidth = headingWidth(block.heading) + HEADING_INSET * 2;
        const padding = FRAME_PADDING;

        return {
            "elk.padding": "[top=40,left=" + padding + ",bottom=" + padding + ",right=" + padding + "]",
            "elk.nodeSize.constraints": "[MINIMUM_SIZE]",
            "elk.nodeSize.minimum": "(" + Math.ceil(minimumWidth) + ", 0)",
        };
    }

    function widest(lines) {
        return lines.reduce((width, line) => Math.max(width, textWidth(line)), 0);
    }

    function shadowOf(box) {
        return box.width > 1 ? SHADOW * 2 : 0;
    }

    function boxSize(box) {
        const shadow = shadowOf(box);

        return {
            width: widest(box.lines) + PADDING_X * 2 + shadow,
            height: box.lines.length * LINE_HEIGHT + PADDING_Y * 2 + shadow,
        };
    }

    function labelsOf(link) {
        if (!diagram.options.show_param_names || link.param === null) {
            return [];
        }

        return [{ text: link.param, width: textWidth(link.param), height: LABEL_HEIGHT }];
    }

    function buildGraph() {
        const children = [];
        const containers = new Map();

        if (diagram.planned) {
            for (const block of diagram.blocks) {
                const node = {
                    id: block.id,
                    children: [],
                    layoutOptions: frameOptions(block),
                };

                containers.set(block.id, node.children);
                children.push(node);
            }
        }

        const containerOf = block => (diagram.planned ? containers.get(block) : children);

        for (const box of diagram.boxes) {
            const size = boxSize(box);
            containerOf(box.block).push({ id: box.id, width: size.width, height: size.height });
        }

        for (const exchange of diagram.exchanges) {
            containerOf(exchange.block).push({ id: exchange.id, width: EXCHANGE_SIZE, height: EXCHANGE_SIZE });
        }

        const edges = diagram.links.map((link, index) => ({
            id: "e" + index,
            sources: [link.from],
            targets: [link.to],
            labels: labelsOf(link),
        }));

        return {
            id: "root",
            layoutOptions: {
                "elk.algorithm": "layered",
                "elk.direction": "RIGHT",
                "elk.hierarchyHandling": "INCLUDE_CHILDREN",
                "elk.edgeRouting": "ORTHOGONAL",
                "elk.layered.spacing.nodeNodeBetweenLayers": "56",
                "elk.spacing.nodeNode": "28",
                "elk.spacing.edgeLabel": "4",
            },
            children,
            edges,
        };
    }

    function element(name, attributes, parent) {
        const node = document.createElementNS(SVG_NS, name);

        for (const [key, value] of Object.entries(attributes)) {
            node.setAttribute(key, value);
        }

        if (parent) {
            parent.appendChild(node);
        }

        return node;
    }

    function label(parent, x, y, content, className) {
        const node = element("text", { x, y, class: className }, parent);
        node.textContent = content;
        return node;
    }

    function tooltip(parent, lines) {
        const node = element("title", {}, parent);
        node.textContent = lines.join("\n");
    }

    function tooltipOf(box) {
        const block = blocksById.get(box.block);
        const lines = [box.name, box.identifier];

        if (box.residency !== null) {
            lines.push("side: " + box.residency);
        }

        if (block.granularity !== null) {
            lines.push("granularity: " + block.granularity);
        }

        lines.push("workers: " + box.width);

        const reason = box.reason !== null ? box.reason : block.reason;
        if (reason !== null) {
            lines.push("narrowed: " + reason);
        }

        return lines;
    }

    function providerOf(layer) {
        return diagram.boxes.find(box => box.provider !== null && box.provider.layers.some(served => served.layer === layer));
    }

    function demandersOf(provider) {
        const layers = new Set(provider.provider.layers.map(served => served.layer));
        return diagram.boxes.filter(box => box.demands.some(demand => layers.has(demand.layer)));
    }

    // Walks the links back from the demanding box, keeping each one on a path that reaches the provider.
    function linksBetween(provider, demander) {
        const found = new Set();
        const reaches = new Map();

        function walk(endpoint) {
            if (endpoint === provider.id) {
                return true;
            }

            if (reaches.has(endpoint)) {
                return reaches.get(endpoint);
            }

            reaches.set(endpoint, false);

            let reached = false;
            diagram.links.forEach((link, index) => {
                if (link.to === endpoint && walk(link.from)) {
                    found.add(index);
                    reached = true;
                }
            });

            reaches.set(endpoint, reached);
            return reached;
        }

        walk(demander.id);
        return found;
    }

    function highlighted(provider, demanders) {
        const elements = [boxGroups.get(provider.id)];

        for (const demander of demanders) {
            elements.push(boxGroups.get(demander.id));

            for (const index of linksBetween(provider, demander)) {
                elements.push(...(linkLines.get(index) || []));
            }
        }

        return elements.filter(node => node !== undefined);
    }

    function highlightOnHover(target, elementsOf) {
        target.addEventListener("mouseenter", () => {
            for (const node of elementsOf()) {
                node.classList.add("highlight");
            }
        });

        target.addEventListener("mouseleave", () => {
            for (const node of elementsOf()) {
                node.classList.remove("highlight");
            }
        });
    }

    function demandHighlight(box, demand) {
        const provider = providerOf(demand.layer);
        return () => (provider === undefined ? [] : highlighted(provider, [box]));
    }

    function providerHighlight(provider) {
        return () => highlighted(provider, demandersOf(provider));
    }

    function drawFrame(parent, block, x, y, width, height) {
        element("rect", { class: "frame", x, y, width, height, rx: 8 }, parent);
        label(parent, x + HEADING_INSET, y + 24, block.heading, "heading");
    }

    function drawBox(parent, box, x, y, width, height) {
        const shadow = shadowOf(box);
        const boxWidth = width - shadow;
        const boxHeight = height - shadow;

        const group = element("g", { class: "box " + box.kind, "data-box": box.id }, parent);
        tooltip(group, tooltipOf(box));
        boxGroups.set(box.id, group);

        if (box.provider !== null) {
            highlightOnHover(group, providerHighlight(box));
        }

        const copies = shadow > 0 ? [2, 1] : [];
        for (const copy of copies) {
            const offset = SHADOW * copy;
            element("rect", { class: "shape shadow", x: x + offset, y: y + offset, width: boxWidth, height: boxHeight, rx: 4 }, group);
        }

        element("rect", { class: "shape", x, y, width: boxWidth, height: boxHeight, rx: 4 }, group);

        // A box's demands are its last lines, one each and in order.
        const firstDemand = box.lines.length - box.demands.length;

        box.lines.forEach((line, index) => {
            const baseline = y + PADDING_Y + LINE_HEIGHT * index + LINE_HEIGHT * 0.72;
            const demand = index >= firstDemand ? box.demands[index - firstDemand] : null;
            const className = index === 0 ? "name" : demand !== null ? "detail demand" : "detail";
            const text = label(group, x + boxWidth / 2, baseline, line, className);

            if (demand !== null) {
                highlightOnHover(text, demandHighlight(box, demand));
            }
        });

        if (box.width > 1) {
            const badge = "×" + box.width;
            const badgeWidth = textWidth(badge) + 10;
            element("rect", { class: "badge", x: x + boxWidth - badgeWidth / 2, y: y - 9, width: badgeWidth, height: 18, rx: 9 }, group);
            label(group, x + boxWidth, y + 4, badge, "badge-text");
        }
    }

    function drawExchange(parent, exchange, x, y, width, height) {
        const group = element("g", { class: "exchange " + exchange.kind }, parent);
        tooltip(group, exchange.lines);

        const middleX = x + width / 2;
        const middleY = y + height / 2;
        const corners = [[middleX, y], [x + width, middleY], [middleX, y + height], [x, middleY]];
        const points = corners.map(corner => corner.join(",")).join(" ");
        element("polygon", { class: "shape", points }, group);

        const ratio = exchange.kind === "disperse" ? "1:" + exchange.width : exchange.width + ":1";
        label(group, middleX, middleY + 4, ratio, "ratio");
    }

    function placeChildren(children, offsetX, offsetY, frames, nodes) {
        for (const child of children) {
            const x = offsetX + child.x;
            const y = offsetY + child.y;

            if (blocksById.has(child.id)) {
                blockOrigins.set(child.id, { x, y });
                drawFrame(frames, blocksById.get(child.id), x, y, child.width, child.height);
                placeChildren(child.children || [], x, y, frames, nodes);
            } else if (boxesById.has(child.id)) {
                drawBox(nodes, boxesById.get(child.id), x, y, child.width, child.height);
            } else if (exchangesById.has(child.id)) {
                drawExchange(nodes, exchangesById.get(child.id), x, y, child.width, child.height);
            }
        }
    }

    function drawEdge(parent, edge) {
        const index = Number(edge.id.slice(1));
        const link = diagram.links[index];
        const kind = link.bounded === null ? "local" : link.bounded ? "bounded" : "unbounded";

        // ELK places an edge relative to the block that contains both of its ends.
        const origin = blockOrigins.get(edge.container) || { x: 0, y: 0 };

        const lines = [];
        linkLines.set(index, lines);

        for (const section of edge.sections || []) {
            const path = [section.startPoint, ...(section.bendPoints || []), section.endPoint];
            const points = path.map(point => (origin.x + point.x) + "," + (origin.y + point.y)).join(" ");
            const attributes = { class: "link " + kind, points, "marker-end": "url(#arrow)", "data-link": edge.id };
            lines.push(element("polyline", attributes, parent));
        }

        for (const edgeLabel of edge.labels || []) {
            const x = origin.x + edgeLabel.x + edgeLabel.width / 2;
            const y = origin.y + edgeLabel.y + edgeLabel.height - 3;
            label(parent, x, y, edgeLabel.text, "link-label");
        }
    }

    function draw(laid) {
        const width = laid.width || 0;
        const height = laid.height || 0;
        const svg = element("svg", { xmlns: SVG_NS, width, height, viewBox: "0 0 " + width + " " + height });

        const defs = element("defs", {}, svg);
        const marker = element("marker", { id: "arrow", viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 7, markerHeight: 7, orient: "auto-start-reverse" }, defs);
        element("path", { d: "M 0 0 L 10 5 L 0 10 z", class: "arrow-head" }, marker);

        const frames = element("g", { class: "frames" }, svg);
        const nodes = element("g", { class: "nodes" }, svg);
        const links = element("g", { class: "links" }, svg);

        placeChildren(laid.children || [], 0, 0, frames, nodes);

        for (const edge of laid.edges || []) {
            drawEdge(links, edge);
        }

        canvas.appendChild(svg);

        if (typeof svgPanZoom !== "undefined") {
            svgPanZoom(svg, { fit: true, center: true, controlIconsEnabled: true, minZoom: 0.1, maxZoom: 10 });
        }
    }

    const elk = new ELK();

    elk.layout(buildGraph())
        .then(draw)
        .catch(error => showMessage("The graph could not be laid out: " + error));
})();
