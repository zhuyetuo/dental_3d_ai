import { chartOrder, cssColor, GUM, Jaw } from "./labels";

export interface OdontogramProps {
  jaw: Jaw;
  /** 每个标签的顶点数，用来区分“存在 / 缺失” */
  counts: Map<number, number>;
  active: number;
  onPick: (fdi: number) => void;
}

/** 牙位图：点选要涂的牙号。屏幕左侧是患者右侧（口腔科惯例）。 */
export function Odontogram({ jaw, counts, active, onPick }: OdontogramProps) {
  const order = chartOrder(jaw);
  const chip = (fdi: number, text: string) => {
    const present = (counts.get(fdi) ?? 0) > 0;
    return (
      <button
        key={fdi}
        type="button"
        className={`chip${fdi === active ? " active" : ""}${present ? "" : " absent"}`}
        style={{ background: cssColor(fdi) }}
        title={present ? `${text}：${counts.get(fdi)} 个顶点` : `${text}：当前没有`}
        onClick={() => onPick(fdi)}
      >
        {text}
      </button>
    );
  };
  return (
    <div className="odontogram">
      <div className="chart-row">
        <span className="side">右</span>
        {order.slice(0, 8).map((f) => chip(f, String(f)))}
        <span className="midline" />
        {order.slice(8).map((f) => chip(f, String(f)))}
        <span className="side">左</span>
      </div>
      <div className="chart-row">{chip(GUM, "牙龈")}</div>
    </div>
  );
}
