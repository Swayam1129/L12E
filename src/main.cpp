// Lab 12: 2D ray tracing with a BVH. Work through the TODOs below and answer the questions in
// written.txt.
// Bonus 2 (fixed): circles are spawned fully inside the screen and bounce off its edges, so none
// get cut off by the window border.

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <optional>
#include <random>
#include <vector>

#include "aabb.h"
#include "vector4.h"
#include "vector_util.h"
#include <SFML/Graphics.hpp>

constexpr int gScreenWidth = 1024;
constexpr int gScreenHeight = 768;
constexpr float EPSILON = 0.001f;

struct Circle {
    constexpr bool intersects(sf::Vector2f point) const {
        return (point - position).lengthSquared() < radius * radius;
    }

    constexpr AABB<float> bounds() const { return {position - radius, position + radius}; }

    sf::Vector2f position;
    float radius;
    Vector4f color;
};

struct BVHNode {
    AABB<float> bounds;
    int childAOrIdx = -1;  // if leaf node, first index into index list, else child bvh node ptr
    int childBOrLen = -1;  // if leaf node, length into index list, else child bvh node ptr
    int isLeaf = 0;
    int height = 0;  // Bonus 1: height in the tree (leaf = 0). Replaces padding, size stays 32.
};
static_assert(sizeof(BVHNode) == 32, "keep BVHNode a power-of-two size");

struct BVH {
    static constexpr float T_ISECT = 1;
    static constexpr float T_TRAV = 1.f / 8;
    static constexpr int BINS = 32;

    explicit BVH(const std::vector<Circle>& prims) : primitives(&prims), indices(prims.size()) {
        std::iota(indices.begin(), indices.end(), 0);
        nodes.reserve(2 * prims.size());
        nodes.push_back({});
        constructNode(0, static_cast<int>(indices.size()), 0);
    }

    // Q4: throw away the old tree and build a brand new one from the current circle positions.
    void reset() {
        nodes.clear();
        nodes.push_back({});
        constructNode(0, static_cast<int>(indices.size()), 0);
    }

    // Builds the node at bvhOffset and returns its height (leaf = 0).
    int constructNode(int startIdx, int numIndices, int bvhOffset) {
        const std::vector<Circle>& prims = *primitives;
        const auto first = indices.begin() + startIdx;
        const auto last = first + numIndices;
        const auto centroid = [&](int idx) { return prims[idx].bounds().centroid(); };

        // determine total bound of centroids (to determine bins)
        AABB<float> centroidBound{};
        for (auto it = first; it != last; ++it) {
            centroidBound |= centroid(*it) + EPSILON;
            centroidBound |= centroid(*it) - EPSILON;
        }
        const sf::Vector2f binWidth =
            (centroidBound.maxPoint - centroidBound.minPoint) / static_cast<float>(BINS);
        assert(binWidth.x != 0);
        assert(binWidth.y != 0);
        const sf::Vector2f invBinWidth = {1.f / binWidth.x, 1.f / binWidth.y};

        // Which bin a primitive falls in along each dimension (0 = x, 1 = y). Clamped so float
        // rounding can never index outside the bin arrays.
        const auto binOf = [&](int idx) {
            const sf::Vector2i bin((centroid(idx) - centroidBound.minPoint) * invBinWidth);
            return sf::Vector2i(std::clamp(bin.x, 0, BINS - 1), std::clamp(bin.y, 0, BINS - 1));
        };

        // find bin each primitive belongs to, per dimension (0 = x, 1 = y)
        std::array<std::array<AABB<float>, BINS>, 2> binBounds{};
        std::array<std::array<int, BINS>, 2> binCounts{};
        for (auto it = first; it != last; ++it) {
            const sf::Vector2i bin = binOf(*it);
            for (int dim : {0, 1}) {
                binBounds[dim][axis(bin, dim)] |= prims[*it].bounds();
                ++binCounts[dim][axis(bin, dim)];
            }
        }
        // prefix and suffix sums
        std::array<std::array<AABB<float>, BINS + 1>, 2> binBoundsPrefix{};
        std::array<std::array<int, BINS + 1>, 2> binCountsPrefix{};
        std::array<std::array<AABB<float>, BINS + 1>, 2> binBoundsSuffix{};
        std::array<std::array<int, BINS + 1>, 2> binCountsSuffix{};
        for (int dim : {0, 1}) {
            for (int i = 0; i < BINS; ++i) {
                binBoundsPrefix[dim][i + 1] = binBoundsPrefix[dim][i] | binBounds[dim][i];
                binCountsPrefix[dim][i + 1] = binCountsPrefix[dim][i] + binCounts[dim][i];
            }
            for (int i = BINS - 1; i >= 0; --i) {
                binBoundsSuffix[dim][i] = binBoundsSuffix[dim][i + 1] | binBounds[dim][i];
                binCountsSuffix[dim][i] = binCountsSuffix[dim][i + 1] + binCounts[dim][i];
            }
        }
        const AABB<float> totalBound = binBoundsPrefix[0].back();
        const float totalPerimLength = totalBound.perimeterLength();

        // Compute best split. Split BINS - 1 (everything on the left) is skipped since it is
        // always worse than just making a leaf node.
        float minCost = std::numeric_limits<float>::infinity();
        int bestDim = -1;
        int bestSplit = -1;
        for (int split = 0; split < BINS - 1; ++split) {
            for (int dim : {0, 1}) {
                const float cost =
                    T_TRAV +
                    T_ISECT *
                        ((binBoundsPrefix[dim][split + 1].perimeterLength() / totalPerimLength) *
                             binCountsPrefix[dim][split + 1] +
                         (binBoundsSuffix[dim][split + 1].perimeterLength() / totalPerimLength) *
                             binCountsSuffix[dim][split + 1]);
                if (cost < minCost) {
                    minCost = cost;
                    bestDim = dim;
                    bestSplit = split;
                }
            }
        }

        const auto makeLeaf = [&] {
            nodes[bvhOffset] = BVHNode{totalBound, startIdx, numIndices, /*isLeaf=*/true, 0};
            return 0;
        };
        // see if we should be leaf node
        if (numIndices * T_ISECT < minCost) return makeLeaf();
        // Repartition indices based on the chosen split. Use the exact same bin computation as
        // above: comparing against a recomputed pivot position can disagree with the binning by
        // float rounding, leaving one side empty, and then the full side recurses forever
        // (stack overflow / segfault once circles move and the tree is rebuilt every frame).
        const auto mid = std::partition(
            first, last, [&](int idx) { return axis(binOf(idx), bestDim) <= bestSplit; });
        const int smallerCount = static_cast<int>(mid - first);
        if (smallerCount == 0 || smallerCount == numIndices) return makeLeaf();
        // internal node; children are appended after their parent
        const int childA = static_cast<int>(nodes.size());
        const int childB = childA + 1;
        nodes[bvhOffset] = BVHNode{totalBound, childA, childB, /*isLeaf=*/false};
        nodes.resize(nodes.size() + 2);  // filled in by the recursive calls
        const int heightA = constructNode(startIdx, smallerCount, childA);
        const int heightB = constructNode(startIdx + smallerCount, numIndices - smallerCount, childB);
        // don't hold a reference across the recursive calls: nodes may have been resized
        nodes[bvhOffset].height = 1 + std::max(heightA, heightB);
        return nodes[bvhOffset].height;
    }

    // Refit every box bottom-up without changing the tree structure. Children are always stored
    // after their parent, so a reverse sweep visits them first.
    void fullRefit() {
        for (int i = static_cast<int>(nodes.size()) - 1; i >= 0; --i) {
            BVHNode& node = nodes[i];
            if (node.isLeaf) {
                node.bounds = {};
                for (int j = node.childAOrIdx; j < node.childAOrIdx + node.childBOrLen; ++j)
                    node.bounds |= (*primitives)[indices[j]].bounds();
            } else {
                node.bounds = nodes[node.childAOrIdx].bounds | nodes[node.childBOrLen].bounds;
            }
        }
    }

    const std::vector<Circle>* primitives;
    std::vector<int> indices;
    std::vector<BVHNode> nodes;
};

// On a hit, color is set to the color of the last circle hit.
bool intersects(sf::Vector2f point, const BVHNode& node, const BVH& bvh, Vector4f& color) {
    if (!node.bounds.intersects(point)) return false;
    if (node.isLeaf) {
        bool hit = false;
        for (int i = node.childAOrIdx; i < node.childAOrIdx + node.childBOrLen; ++i) {
            const Circle& circle = (*bvh.primitives)[bvh.indices[i]];
            if (circle.intersects(point)) {
                color = circle.color;
                hit = true;
            }
        }
        return hit;
    }
    // evaluate both children (no short-circuit) so the overlap color stays consistent
    const bool hitA = intersects(point, bvh.nodes[node.childAOrIdx], bvh, color);
    const bool hitB = intersects(point, bvh.nodes[node.childBOrLen], bvh, color);
    return hitA || hitB;
}

// SOA layout for easy buffering
struct Scene {
    std::size_t createCircle(sf::Vector2f position, float radius, const Vector4f& color,
                             sf::Vector2f movement = {}) {
        circles.push_back({position, radius, color});
        movements.push_back(movement);
        return circles.size() - 1;
    }

    // Q3: move every circle by its movement (pixels per frame).
    // Bonus 2: bounce off the screen edges so a circle never leaves the window.
    void update() {
        for (std::size_t i = 0; i < circles.size(); ++i) {
            Circle& c = circles[i];
            sf::Vector2f& m = movements[i];
            c.position += m;
            if (c.position.x - c.radius < 0) { c.position.x = c.radius; m.x = -m.x; }
            if (c.position.x + c.radius > gScreenWidth) { c.position.x = gScreenWidth - c.radius; m.x = -m.x; }
            if (c.position.y - c.radius < 0) { c.position.y = c.radius; m.y = -m.y; }
            if (c.position.y + c.radius > gScreenHeight) { c.position.y = gScreenHeight - c.radius; m.y = -m.y; }
        }
    }

    Circle& circle(std::size_t i) {
        assert(valid(i));
        return circles[i];
    }

    bool valid(std::size_t i) const { return i < circles.size(); }

    std::vector<Circle> circles;
    std::vector<sf::Vector2f> movements;  // movement of circles[i], parallel array
};

void Draw(sf::RenderWindow& window, const AABB<float>& aabb,
          sf::Color outline = sf::Color::White) {
    // draw frame of rectangle
    static sf::RectangleShape shape;
    shape.setSize(aabb.maxPoint - aabb.minPoint);
    shape.setPosition(aabb.minPoint);
    shape.setFillColor(sf::Color::Transparent);
    shape.setOutlineColor(outline);
    shape.setOutlineThickness(1.0f);
    window.draw(shape);
}

// Bonus 1: blue for nodes just above the leaves, through green, to red for the root.
sf::Color HeightColor(int height, int maxHeight) {
    const float t = maxHeight > 1 ? static_cast<float>(height - 1) / (maxHeight - 1) : 1.f;
    const auto lerp = [](float a, float b, float t) { return static_cast<std::uint8_t>(a + (b - a) * t); };
    if (t < 0.5f) return {0, lerp(80, 255, 2 * t), lerp(255, 0, 2 * t)};  // blue -> green
    return {lerp(0, 255, 2 * t - 1), lerp(255, 40, 2 * t - 1), 0};        // green -> red
}

int main() {
    const sf::Vector2u screenSize(gScreenWidth, gScreenHeight);
    sf::RenderWindow window(sf::VideoMode(screenSize), "BVH", sf::State::Windowed);
    window.setFramerateLimit(30);

    std::mt19937 rng;  // default seed: the same scene every run
    const auto uniform = [&rng](float lo, float hi) {
        return std::uniform_real_distribution<float>(lo, hi)(rng);
    };
    Scene scene;
    constexpr float PI = 3.14159265f;
    for (int n = 0; n < 100; ++n) {
        const float radius = uniform(10, 20);
        // Bonus 2: keep the whole circle on screen instead of just its centre
        const sf::Vector2f position{uniform(radius, gScreenWidth - radius),
                                    uniform(radius, gScreenHeight - radius)};
        const Vector4f color{uniform(0, 1), uniform(0, 1), uniform(0, 1), 1};
        const float angle = uniform(0, 2 * PI);
        const float speed = uniform(1, 4);
        scene.createCircle(position, radius, color,
                           {speed * std::cos(angle), speed * std::sin(angle)});
    }
    BVH bvh(scene.circles);

    std::vector<std::uint8_t> pixels(4 * gScreenWidth * gScreenHeight);  // RGBA, starts black
    sf::Texture texture(screenSize);
    const sf::Sprite sprite(texture);

    // Q3 uses fullRefit() every frame; Q4 uses reset(). Press R to switch between them.
    bool rebuildEachFrame = true;

    while (window.isOpen()) {
        while (const std::optional event = window.pollEvent()) {
            if (event->is<sf::Event::Closed>())
                window.close();
            else if (const auto* key = event->getIf<sf::Event::KeyPressed>())
            {
                if (key->code == sf::Keyboard::Key::Escape) window.close();
                if (key->code == sf::Keyboard::Key::R) rebuildEachFrame = !rebuildEachFrame;
            }
        }

        scene.update();
        if (rebuildEachFrame)
            bvh.reset();  // Q4: rebuild the tree from scratch
        else
            bvh.fullRefit();  // Q3: keep the tree, only grow/shrink its boxes

        for (int y = 0; y < gScreenHeight; ++y) {
            for (int x = 0; x < gScreenWidth; ++x) {
                const sf::Vector2f pixel(static_cast<float>(x), static_cast<float>(y));
                // Q2: antialiasing, 4 samples at (+-1/3, +-1/3) averaged
                constexpr float OFF = 0.3333f;
                constexpr std::array<sf::Vector2f, 4> offsets{
                    {{-OFF, -OFF}, {OFF, -OFF}, {-OFF, OFF}, {OFF, OFF}}};
                Vector4f color{0, 0, 0, 0};
                for (const sf::Vector2f& offset : offsets) {
                    Vector4f sample{0, 0, 0, 0};  // stays transparent black on a miss
                    intersects(pixel + offset, bvh.nodes[0], bvh, sample);
                    color += sample;
                }
                color /= static_cast<float>(offsets.size());
                const std::size_t i = 4 * (static_cast<std::size_t>(y) * gScreenWidth + x);
                pixels[i + 0] = static_cast<std::uint8_t>(255 * color.x);  // R
                pixels[i + 1] = static_cast<std::uint8_t>(255 * color.y);  // G
                pixels[i + 2] = static_cast<std::uint8_t>(255 * color.z);  // B
                pixels[i + 3] = static_cast<std::uint8_t>(255 * color.w);  // A
            }
        }

        texture.update(pixels.data());

        window.clear();
        window.draw(sprite);
        // Q1: only draw internal (non-leaf) nodes; Bonus 1: color them by height
        const int maxHeight = bvh.nodes[0].height;
        for (const BVHNode& node : bvh.nodes) {
            if (node.isLeaf) continue;
            Draw(window, node.bounds, HeightColor(node.height, maxHeight));
        }
        window.display();
    }
    return 0;
}
